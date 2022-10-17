from collections import deque, defaultdict
import os
import logging
import time
import json
import gym
import torch.nn as nn
import torch
import numpy as np
import pickle
import skfmm
import skimage.morphology
from numpy import ma

from agent.mapping import Semantic_Mapping
from agent.prediction import SemOccPred
from constants import habitat_goal_label_to_similar_coco, twentyone_to_hm3d
from arguments import get_args
import pickle


def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h + 2, w + 2)) + value
            new_mat[1:h + 1, 1:w + 1] = mat
            return new_mat

class Agent_State:
    def __init__(self,args):
        self.args = args
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        self.nc = 4 + args.num_sem_categories # num channels
        self.gt_mask_channel = 8
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
        self.device = args.device = torch.device("cuda:" + str(args.sem_gpu_id) if args.cuda else "cpu")

        self.map_size = args.map_size_cm // args.map_resolution
        self.full_w, self.full_h = self.map_size, self.map_size
        self.local_w = int(self.full_w / args.global_downscaling)
        self.local_h = int(self.full_h / args.global_downscaling)
        self.grid_w, self.grid_h = self.full_w//args.grid_resolution, self.full_h//args.grid_resolution
        self.grid_nc = args.record_frames + args.record_angle + 1 + 1 # blacklist and max score


        # In the actual implementation, we have two maps with different granularities to save time
        # self.grid stores information related to object goal identification and has bigger grid,
        # since we need to run flood-fill algorithm on it and we want to save time

        # self.full_map stores both obstacle related information and goal related information.
        # it has smaller grid since it is used in path planning and requires higher precision.

        # Dimensions
        # 0: number of frames, Channel 4 in the paper
        # 1: cumulative confidence, Channel 5 in the paper
        # 2, 3: not used in final implementation
        # 5: max confidence, Channel 6 in the paper.
        self.grid = torch.zeros(self.grid_nc,self.grid_w,self.grid_h).float().to(self.device)
        self.local_grid_w =self.grid_w // args.global_downscaling
        self.local_grid_h = self.local_grid_w
        self.local_grid = torch.zeros(self.grid_nc, self.local_grid_w,
                                      self.local_grid_h).float().to(self.device)


        # Initializing full and local map
        # dimensions:
        # 0: depth collision, corresponding to Channel 1 in the paper
        # 1: exploration coverage
        # 4: goal category, corresponding to Channel 6 in the paper
        # 5: conflict category, corresponding to Channel 7 in the paper
        # The rest of the dimensions of self.full_map are not used in the final implementation
        self.full_map = torch.zeros(self.nc, self.full_w, self.full_h).float().to(
            self.device)

        self.local_map = torch.zeros(self.nc, self.local_w,
                                self.local_h).float().to(self.device)
        self.global_goal_loc = torch.zeros(self.full_w,self.full_h)
        self.global_goal_index = (-1,-1)

        # Initial full and local pose
        self.full_pose = torch.zeros( 3).float().to(self.device)
        self.local_pose = torch.zeros( 3).float().to(self.device)

        # Origin of local map
        self.origins = np.zeros((3))

        # Local Map Boundaries
        self.lmb = np.zeros(( 4)).astype(int)

        # Planner pose inputs has 7 dimensions
        # 1-3 store continuous global agent location
        # 4-7 store local map boundaries
        self.planner_pose_inputs = np.zeros(( 7))

        # Global policy observation space
        self.es = 2
        self.log_history = []
        self.goal_log = []
        self.hard_goal = False

        # Semantic Mapping
        self.sem_map_module = Semantic_Mapping(args).to(self.device)
        self.sem_map_module.eval()
        self.global_goal_preset = (0.1,0.1)
        self.global_goal_rotation = [(0.1,0.1),(0.9,0.1),(0.9,0.9),(0.1,0.9)]
        self.global_goal_rotation_id = 0
        self.cat_semantic_map = torch.zeros(self.local_w, self.local_h)
        self.local_grid_vis = torch.zeros(self.local_grid_w,self.local_grid_h)
        self.score_threshold = None

        self.stuck = False
        if args.detect_stuck == 1:
            self.pos_record = []

        # Semantic Occupancy Prediction
        self.sem_occ_pred = SemOccPred(args) if args.only_explore == 0 else None 
        self.selem = skimage.morphology.disk(args.col_rad - args.dd_erode)
        self.selem4idx = np.where(skimage.morphology.disk(args.col_rad + 1) > 0)
        self.so_pred = None
        self.value = None
        self.dd_wt = None
        self.dd_wt1 = None
        self.last_global_goal = None
        
        self.oventime = False

    def reset(self):
        self.l_step = 0
        self.g_step = 0
        self.step = 0
        self.avg_goal_conf = 0
        self.avg_conf_conf = 0
        self.num_conf = 0
        self.goal_cat = -1
        self.found_goal = False
        self.hard_goal = False
        self.global_goal_rotation_id = 0
        self.global_goal_preset = (0.1,0.1)
        self.goal_log = []
        self.score_threshold = 0.85
        self.init_map_and_pose()
        self.stuck = False
        self.so_pred = None
        self.value = None
        self.dd_wt = None
        self.dd_wt1 = None
        self.oventime = False
        self.last_global_goal = None
        if self.args.detect_stuck == 1:
            self.pos_record = []


    def save_conf_stat(self,suc,epi_length, epi_ID,gt_found = False,step = -1):
        if self.num_conf == 0:
            self.avg_goal_conf = -1
            self.avg_conf_conf = -1
        else:
            self.avg_goal_conf /= self.num_conf
            self.avg_conf_conf /= self.num_conf
        self.log_history.append({
            "suc":suc,
            "epi_len":epi_length,
            "epi_ID":epi_ID,
            "goal":self.goal_cat,
            "self_suc":self.found_goal,
            "goal_score":self.avg_goal_conf,
            "conf_score":self.avg_conf_conf,
            "gt_found": gt_found,
            "goal_log": self.goal_log,
            "stuck": self.stuck,
            "step_found":step
        })
        with open(self.args.log_path, 'wb') as handle:
            pickle.dump(self.log_history, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if self.num_conf == 0:
            return



    def inc_step(self):
        args = self.args
        self.l_step += 1
        self.step += 1
        self.l_step = self.step % args.num_local_steps
    def init_with_obs(self,obs,infos):

        self.l_step = 0
        self.step = 0

        self.poses = torch.from_numpy(np.asarray(
            infos['sensor_pose'])
        ).float().to(self.device)
        _, self.local_map, _, self.local_pose = \
            self.sem_map_module(obs, self.poses, self.local_map, self.local_pose,self)

        # Compute Global policy input
        self.locs = self.local_pose.cpu().numpy()

        r, c = self.locs[1], self.locs[0]
        loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
                        int(c * 100.0 / self.args.map_resolution)]

        self.local_map[2:4, loc_r - 1:loc_r + 2,
        loc_c - 1:loc_c + 2] = 1.
        
        # ang = np.random.uniform(low=-np.pi, high=np.pi)
        # rgoal = [np.cos(ang) * 0.25 + 0.5, np.sin(ang) * 0.25 + 0.5]
        rgoal = [0.1, 0.1]
        self.global_goals = [[int(rgoal[0] * self.local_w), int(rgoal[1] * self.local_h)]]
        self.global_goals = [[min(x, int(self.local_w - 1)), min(y, int(self.local_h - 1))]
                        for x, y in self.global_goals]

        self.goal_maps = np.zeros((self.local_w, self.local_h))


        self.goal_maps[self.global_goals[0][0], self.global_goals[0][1]] = 1


        p_input = {}

        p_input['map_pred'] = self.local_map[0, :, :].cpu().numpy()
        p_input['exp_pred'] = self.local_map[1, :, :].cpu().numpy()
        p_input['pose_pred'] = self.planner_pose_inputs
        p_input['goal'] = self.goal_maps  # global_goals[e]
        p_input['new_goal'] = 1
        p_input['found_goal'] = 0
        p_input['wait'] = 0  # does it matter?
        if self.args.visualize or self.args.print_images:
            vlm = torch.clone(self.local_map[4:, :, :])
            vlm[-1] = 1e-5
            p_input['sem_map_pred'] = vlm.argmax(0).cpu().numpy()


        self.planner_inputs = p_input


        torch.set_grad_enabled(False)


    def get_local_map_boundaries(self, agent_loc, local_sizes, full_sizes):
        loc_r, loc_c = agent_loc
        local_w, local_h = local_sizes
        full_w, full_h = full_sizes

        if self.args.global_downscaling > 1:
            gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
            gx1, gy1 = gx1 - gx1%self.args.grid_resolution, gy1 - gy1%self.args.grid_resolution
            gx2, gy2 = gx1 + local_w, gy1 + local_h
            if gx1 < 0:
                gx1, gx2 = 0, local_w
            if gx2 > full_w:
                gx1, gx2 = full_w - local_w, full_w

            if gy1 < 0:
                gy1, gy2 = 0, local_h
            if gy2 > full_h:
                gy1, gy2 = full_h - local_h, full_h
        else:
            gx1, gx2, gy1, gy2 = 0, full_w, 0, full_h

        return [gx1, gx2, gy1, gy2]

    def init_map_and_pose(self):
        args = self.args
        self.full_map.fill_(0.)
        self.full_pose.fill_(0.)
        self.global_goal_loc.fill_(0.)


        if self.args.record_angle + self.args.record_frames == 4:
            self.grid[0:2].fill_(0.)
            self.grid[2].fill_(6.28)
            self.grid[3].fill_(0.)
            self.grid[4:6].fill_(0.)

            self.local_grid[0:2].fill_(0.)
            self.local_grid[2].fill_(6.28)
            self.local_grid[3].fill_(0.)
            self.local_grid[4:6].fill_(0.)



        self.full_pose[:2] = self.args.map_size_cm / 100.0 / 2.0

        locs = self.full_pose.cpu().numpy()
        self.planner_pose_inputs[:3] = locs

        r, c = locs[1], locs[0]
        loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                        int(c * 100.0 / args.map_resolution)]

        self.full_map[2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

        self.lmb = self.get_local_map_boundaries((loc_r, loc_c),
                                                 (self.local_w, self.local_h),
                                                 (self.full_w, self.full_h))

        self.planner_pose_inputs[3:] = self.lmb
        self.origins = np.array([self.lmb[2] * args.map_resolution / 100.0,
                        self.lmb[0] * args.map_resolution / 100.0, 0.])

        self.local_map = self.full_map[:,
                         self.lmb[0]:self.lmb[1],
                         self.lmb[2]:self.lmb[3]]
        self.local_pose = self.full_pose - \
                          torch.from_numpy(self.origins).to(self.device).float()

    def set_hard_goal(self):
        self.hard_goal = True

    def recur_fill(self,i,j):
        if i < 0 or j < 0 or i >= self.full_w or j >= self.full_h:
            return
        if self.global_goal_loc[i,j] != 0 or self.full_map[4,i,j] <= 0.1:
            return
        self.global_goal_loc[i,j] = 1
        for di in range(-1,1):
            for dj in range(-1,1):
                self.recur_fill(i+di,j+dj)

    def recur_fill_grid(self,i,j):
        if i < 0 or j < 0 or i >= self.local_grid_w or j >= self.local_grid_h:
            return True
        if self.local_grid_vis[i,j] != 0 or self.local_grid[5,i,j] < self.score_threshold-0.01: # I feel like it's the equal sign here
            return True
        if self.local_grid[4,i,j] == 1:
            return False
        self.local_grid_vis[i,j] = 1
        for di in range(-1,1):
            for dj in range(-1,1):
                if self.recur_fill(i+di,j+dj) == False:
                    self.local_grid[4,i,j] = 1
                    return False
        r1, r2 = i * self.args.grid_resolution, (
                i + 1) * self.args.grid_resolution
        c1, c2 = j * self.args.grid_resolution, (
                j + 1) * self.args.grid_resolution
        self.cat_semantic_map[r1:r2, c1:c2] = self.local_map[ 4, r1:r2, c1:c2]
        return True




    def save_global_goal(self):
        self.found_goal = True
        self.full_map[ 4, self.lmb[ 0]:self.lmb[1],
        self.lmb[2]:self.lmb[3]] = \
            self.local_map[4]
        max_index = np.unravel_index(torch.argmax(self.full_map[ 4, :, :]).cpu().numpy(),
                                     (self.full_w,self.full_h))
        self.global_goal_index = max_index
        self.recur_fill(max_index[0],max_index[1])

    # it has to be scattered large
    def global_to_local(self):
        r1,r2,c1,c2 = self.lmb[0],self.lmb[1],self.lmb[2],self.lmb[3]
        r,c = self.global_goal_index
        goal_maps = np.zeros((self.local_w, self.local_h))
        if r1<=r and r <r2 and c1<=c and c<c2:
            return self.global_goal_loc[r1:r2,c1:c2]
        lr,lc = 0,0
        if r < r1:
            lr = 0
        elif r > r2:
            lr = self.local_w-1
        else:
            lr = r-r1

        if c < c1:
            lc = 0
        elif c > c2:
            lc = self.local_h-1
        else:
            lc = c-c1

        siz = 20
        lr1 = max(0,lr-siz)
        lr2 = min(lr+siz,self.local_w-1)
        lc1 = max(0,lc-siz)
        lc2 = min(lc+siz,self.local_h-1)
        goal_maps[lr1:lr2,lc1:lc2] = 1
        return goal_maps

    def suc_gt_map(self,goalmap,gtmap):
        ind = np.nonzero(goalmap)
        siz = len(ind[0])
        gtsiz = np.sum(gtmap[ind])
        if gtsiz > siz*0.6:
            return True
        else:
            return False

    def get_conflict(self,goalmap,gtmap):
        ind = np.nonzero(goalmap)
        siz = len(ind[0])
        gtsiz = np.sum(gtmap[ind])
        if gtsiz > siz*0.5:
            return np.max(gtmap[ind])
        else:
            return 0

    def get_black_white_conflict(self,goalmap,gtmap):
        ind = np.nonzero(goalmap)
        return np.max(gtmap[ind])

    def clear_goal(self,goalmap):
        #this is local
        self.found_goal = False
        ind = np.nonzero(goalmap)
        ind2 = (ind[0] // self.args.grid_resolution,ind[1]//self.args.grid_resolution)
        self.local_grid[4][ind2] = 1

    def goal_record(self,goalmap):
        ind = np.nonzero(goalmap)
        ind2 = (ind[0] // self.args.grid_resolution,
                ind[1] // self.args.grid_resolution)
        ans = {}
        max_score = -1
        max_cumu = -1
        max_ratio = -1

        #max score
        for i in range(len(ind2[0])):
            n,m = ind2[0][i],ind2[1][i]
            if self.local_grid[5,n,m] > max_score:
                max_score = self.local_grid[5,n,m]
                ans["score"] = self.local_grid[:,n,m].cpu().numpy()

        #highest cumu
        for i in range(len(ind2[0])):
            n,m = ind2[0][i],ind2[1][i]
            if self.local_grid[1,n,m] > max_cumu:
                max_cumu = self.local_grid[1,n,m]
                ans["cumu"] = self.local_grid[:,n,m].cpu().numpy()

        #highest ratio
        for i in range(len(ind2[0])):
            n,m = ind2[0][i],ind2[1][i]
            if self.local_grid[1,n,m]/self.local_grid[0,n,m] > max_ratio:
                max_ratio = self.local_grid[1,n,m]/self.local_grid[0,n,m]
                ans["ratio"] = self.local_grid[:,n,m].cpu().numpy()
        ans["total"] = {"score":float(max_score.cpu().numpy()),"cumu":float(max_cumu.cpu().numpy()),"ratio":float(max_ratio.cpu().numpy())}
        ans["suc"] = self.suc_gt_map(goalmap,self.local_map[self.gt_mask_channel,:,:].cpu().numpy()) if self.args.use_gt_mask else -1
        ans["step"] = self.step
        if self.args.record_conflict == 1:
            ans["conflict"] = {"normal":self.get_conflict(goalmap,self.local_map[5,:,:].cpu().numpy()),
                               "black":self.get_black_white_conflict(goalmap,self.local_map[6,:,:].cpu().numpy()),
                               "white": self.get_black_white_conflict(goalmap,
                                                                      self.local_map[
                                                                       7, :,
                                                                      :].cpu().numpy())}

        return ans

    def clear_goal_and_set_gt_map(self,goalmap):
        self.clear_goal(goalmap)
        self.goal_log.append(self.goal_record(goalmap))


    def upd_agent_state(self,obs,infos):

        args = self.args
        self.goal_cat = infos['goal_cat_id']
        self.poses = torch.from_numpy(np.asarray(
            infos['sensor_pose'] )
        ).float().to(self.device)
        
        _, self.local_map, _, self.local_pose = \
            self.sem_map_module(obs, self.poses, self.local_map, self.local_pose,self)
        
        locs = self.local_pose.cpu().numpy()
        self.planner_pose_inputs[:3] = locs + self.origins
        self.local_map[2, :, :].fill_(0.)  # Resetting current location channel

        r, c = locs[1], locs[0]
        loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                        int(c * 100.0 / args.map_resolution)]
        self.local_map[2:4, loc_r - 2:loc_r + 3, loc_c - 2:loc_c + 3] = 1.
        
        # Explored under the agent
        to_fill = (self.selem4idx[0] - (args.col_rad+1) + loc_r, self.selem4idx[1] - (args.col_rad+1) + loc_c)
        self.local_map[1][to_fill] = 1.
        
        dist_to_goal = np.sqrt((loc_r - (self.global_goals[0][0]))**2 + (loc_c - (self.global_goals[0][1]))**2)
        if dist_to_goal < 15:
            to_fill = (self.selem4idx[0] - (args.col_rad+1) + self.global_goals[0][0], 
                       self.selem4idx[1] - (args.col_rad+1) + self.global_goals[0][1])
            self.local_map[1][to_fill] = 1.
                
        if self.args.detect_stuck == 1:
            glo_r, glo_c = self.lmb[0] + loc_r, self.lmb[2] + loc_c
            self.pos_record.append((glo_r, glo_c))
            l = len(self.pos_record)
            if len(self.pos_record) > 110:
                stuck = True
                for i in range(2, 100):
                    dis = abs(glo_r - self.pos_record[l - i][0]) + abs(glo_c - self.pos_record[l - i][1])
                    if dis > 25:
                        stuck = False
                if stuck:
                    self.stuck = True

        if self.l_step == args.num_local_steps - 1:
            self.l_step = 0
            # For every global step, update the full and local maps

            self.full_map[:, self.lmb[0]:self.lmb[1], self.lmb[2]:self.lmb[3]] = \
                self.local_map
            res = self.args.grid_resolution

            # self.grid[:, self.lmb[0] // res:self.lmb[1] // res, self.lmb[2] // res: self.lmb[3] // res] = \
            #     torch.clone(self.local_grid)
            self.full_pose = self.local_pose + \
                             torch.from_numpy(self.origins).to(self.device).float()

            locs = self.full_pose.cpu().numpy()
            r, c = locs[1], locs[0]
            loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                            int(c * 100.0 / args.map_resolution)]

            self.lmb = self.get_local_map_boundaries((loc_r, loc_c),
                                                     (self.local_w, self.local_h),
                                                     (self.full_w, self.full_h))

            self.planner_pose_inputs[3:] = self.lmb
            self.origins = np.array([self.lmb[2] * args.map_resolution / 100.0,
                            self.lmb[0] * args.map_resolution / 100.0, 0.])

            self.local_map = self.full_map[:,
                             self.lmb[0]:self.lmb[1],
                             self.lmb[2]:self.lmb[3]]
            self.local_pose = self.full_pose - \
                              torch.from_numpy(self.origins).to(self.device).float()
            # self.local_grid = torch.clone(self.grid[:, self.lmb[0] // res:self.lmb[1] // res,
            #                               self.lmb[2] // res: self.lmb[3] // res])


            locs = self.local_pose.cpu().numpy()
            r, c = locs[1], locs[0]
            loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                            int(c * 100.0 / args.map_resolution)]
            if self.step < args.switch_step:
                if self.hard_goal:
                    self.global_goal_rotation_id = (self.global_goal_rotation_id + 1)%4
                    self.global_goal_preset = self.global_goal_rotation[self.global_goal_rotation_id]
                    self.hard_goal = False

                # ang = np.random.uniform(low=-np.pi, high=np.pi)
                # self.global_goal_preset = [np.cos(ang) * 0.25 + 0.5, np.sin(ang) * 0.25 + 0.5]

                self.global_goals = [[int(self.global_goal_preset[0] * self.local_w),
                                 int(self.global_goal_preset[1] * self.local_h)]
                                ]
                self.global_goals = [[min(x, int(self.local_w - 1)),
                                 min(y, int(self.local_h - 1))]
                                for x, y in self.global_goals]


        
        # ------------------------------------------------------------------
        
        # Activating prediction
        if (self.step % args.smp_step == args.smp_step - 1 or dist_to_goal < 15) and self.step >= args.switch_step:
            
            self.full_map[:, self.lmb[0]:self.lmb[1], self.lmb[2]:self.lmb[3]] = \
                    self.local_map
            # Extract the prediction in the local map bounds
            if args.num_sem_categories == 16 and self.full_w == 960:
                so_pred = self.sem_occ_pred.get_prediction(self.full_map[:, 120:840, 120:840].cpu().numpy())
                blank = np.zeros((6, 960, 960))
                blank[:, 120:840, 120:840] = so_pred
                so_pred = blank
            else:
                so_pred = self.sem_occ_pred.get_prediction(self.full_map.cpu().numpy())
                
            
            if args.num_sem_categories == 23:
                target = twentyone_to_hm3d[self.goal_cat]
            else:
                target = self.goal_cat
            so_pred = so_pred[target,
                              self.lmb[0]:self.lmb[1],
                              self.lmb[2]:self.lmb[3]]

            # Weight value based on inverse geodesic distance
            trav = skimage.morphology.binary_dilation(np.rint(self.full_map[0].cpu().numpy()), self.selem) != True
            gx1, gx2, gy1, gy2 = int(self.lmb[0]), int(self.lmb[1]), int(self.lmb[2]), int(self.lmb[3])
            
            trav[self.helper.collision_map == 1] = 0
            
            # if args.escape:
            #     traversible_ma = ma.masked_values(trav * 1, 0)
            #     traversible_ma[np.clip(loc_r + self.lmb[0], a_min=0, a_max=self.full_w-1), 
            #                    np.clip(loc_c + self.lmb[2], a_min=0, a_max=self.full_h-1)] = 0
            #     dd = skfmm.distance(traversible_ma, dx=1)
            #     dd = ma.filled(dd, np.max(dd) + 1)
            #     dd[np.where(dd == np.max(dd))] = np.inf
            #     #dd[np.where(dd < 100)] = np.inf
            #     #dd_wt = 20./ (np.clip(dd, a_min=20, a_max=None)) 
            #     self.dd_wt1 = np.exp(-dd / args.alpha)[self.lmb[0]:self.lmb[1], self.lmb[2]:self.lmb[3]]
            
            trav[self.helper.visited_vis == 1] = 1
            
#             trav[0] = 1
#             trav[-1] = 1
#             trav[:, 0] = 1
#             trav[:, -1] = 1
            
            traversible_ma = ma.masked_values(trav * 1, 0)
            traversible_ma[np.clip(loc_r + self.lmb[0], a_min=0, a_max=self.full_w-1), 
                           np.clip(loc_c + self.lmb[2], a_min=0, a_max=self.full_h-1)] = 0
            dd = skfmm.distance(traversible_ma, dx=1)
            dd = ma.filled(dd, np.max(dd) + 1)
            dd[np.where(dd == np.max(dd))] = np.inf
            #dd[np.where(dd < 100)] = np.inf
            #dd_wt = 20./ (np.clip(dd, a_min=20, a_max=None)) 
            dd_wt = np.exp(-dd / args.alpha)[self.lmb[0]:self.lmb[1], self.lmb[2]:self.lmb[3]]
            
            if np.sum(dd_wt) < 10 and self.dd_wt is not None:
                dd_wt = self.dd_wt
            value = so_pred * dd_wt
            self.dd_wt = dd_wt
            self.so_pred = so_pred
            self.value = value
            
            # if self.step % 50 == 49 and args.print_images:
            #     np.save('data/tmp/so_pred%03d.npy' % self.step, so_pred)
            #     np.save('data/tmp/dd%03d.npy' % self.step, dd)
            #     np.save('data/tmp/ddwt%03d.npy' % self.step, 1/ dd_wt)
            #     np.save('data/tmp/value%03d.npy' % self.step, value)
            new_gg = [np.unravel_index(value.argmax(), value.shape)]
            if new_gg != self.last_global_goal:
                self.last_global_goal = self.global_goals
                self.global_goals = new_gg
        # ------------------------------------------------------------------
        
        blind = False
        if args.escape and self.dd_wt is not None:
            if self.step >= 250 and self.step % 25 == 0 and np.sum(self.dd_wt > 0) < args.escape: # and self.local_map[1, :, :].sum() < 4000
                # trapped in stairs?
                self.local_map[0] = 0
                self.full_map[0] = 0
                # self.global_goals = [[20, 20]]
                # blind = True
            
            
        # Update long-term goal if target object is found
        found_goal = 0
        goal_maps = np.zeros((self.local_w, self.local_h))


        goal_maps[self.global_goals[0][0], self.global_goals[0][1]] = 1

        maxi = 0.0
        maxc = -1
        maxa = 0.0
        
#         e = 0
#         cn = 4
#         # use the grid to determine global goal
#         # requires >= 0.75 on the cumulated channel (channel 1)
#         # now change to >= 0.85 on the single channel (channel 5)
#         if self.args.only_explore == 0:
#             max_score = torch.max(self.local_grid[5][self.local_grid[4] == 0])
#             if max_score > self.score_threshold:
#                 indices = torch.nonzero(self.local_grid[5] >= max_score)
#                 self.cat_semantic_map.fill_(0.)
#                 self.local_grid_vis.fill_(0.)
#                 for index in indices:
#                     r1 = index[0] * self.args.grid_resolution
#                     c1 = index[1] * self.args.grid_resolution
#                     r2 = r1 + self.args.grid_resolution
#                     c2 = c1 + self.args.grid_resolution
#                     if self.local_grid[4, index[0], index[1]] == 0:
#                         if self.recur_fill_grid(index[0], index[1]):
#                             found_goal = 1

#                 if found_goal == 1:
#                     self.found_goal = True
#                     cat_semantic_scores = self.cat_semantic_map.cpu().numpy()
#                     cat_semantic_scores[
#                         cat_semantic_scores < self.score_threshold - 0.01] = 0.
#                     goal_maps = cat_semantic_scores

        
        if self.args.only_explore == 0:
            cn = self.goal_cat + 4
            if self.local_map[cn, :, :].sum() != 0.:
                cat_semantic_map = self.local_map[cn, :, :].cpu().numpy()
                cat_semantic_scores = cat_semantic_map
                cat_semantic_scores[cat_semantic_scores > 0] = 1.
                temp_goal = cat_semantic_scores
                if (self.goal_cat != 5 and not self.oventime) and args.num_sem_categories <= 16:  # don't erode TV
                    toilet_plant_reduction = 1 if self.goal_cat == 4 or self.goal_cat == 2 else 0
                    for _ in range(self.args.goal_erode - toilet_plant_reduction):
                        temp_goal = skimage.morphology.binary_erosion(temp_goal.astype(bool)).astype(float)
                    temp_goal = skimage.morphology.binary_dilation(temp_goal.astype(bool)).astype(float)
                    
                if (self.goal_cat != 14) and args.num_sem_categories == 23:  # don't erode TV
                    toilet_reduction = 1 if self.goal_cat == 11 else 0
                    for _ in range(self.args.goal_erode - toilet_reduction):
                        temp_goal = skimage.morphology.binary_erosion(temp_goal.astype(bool)).astype(float)
                    temp_goal = skimage.morphology.binary_dilation(temp_goal.astype(bool)).astype(float)
                    
                if temp_goal.sum() == 0.:
                    temp_goal = cat_semantic_scores
                if args.num_sem_categories != 23:
                    if self.goal_cat == 3:  # bed vs sofa, chair, table
                        temp_goal *= self.local_map[4 + 1, :, :].cpu().numpy() == 0
                        temp_goal *= self.local_map[4 + 0, :, :].cpu().numpy() == 0
                        temp_goal *= self.local_map[4 + 6, :, :].cpu().numpy() == 0
                    if self.goal_cat == 1:  # sofa vs chair
                        temp_goal *= self.local_map[4 + 0, :, :].cpu().numpy() == 0
                    # if self.goal_cat == 5:  # TV vs oven
                    #     temp_goal *= self.local_map[4 + 7, :, :].cpu().numpy() == 0
                else:
                    if self.goal_cat == 7:  # bed vs sofa
                        temp_goal *= self.local_map[4 + 6, :, :].cpu().numpy() == 0
                if temp_goal.sum() != 0.:
                    goal_maps = temp_goal
                    found_goal = 1
                    
        if args.num_sem_categories != 23 and args.oventrick:
            if self.step > 400 and self.goal_cat == 5 and found_goal == 0:  # bring in the ovens for TV
                self.local_map[self.goal_cat + 4, :, :] += self.local_map[7 + 4, :, :]
                self.oventime = True
        # else:
        #     if self.step > 400 and self.goal_cat == 14 and found_goal == 0:  # bring in the ovens for TV
        #         self.local_map[self.goal_cat + 4, :, :] += self.local_map[18 + 4, :, :]
        #         self.oventime = True
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Take action and get next observation
        p_input = {}

        p_input['map_pred'] = self.local_map[0, :, :].cpu().numpy()
        if blind:
            p_input['map_pred'][:] = 0
        p_input['exp_pred'] = self.local_map[1, :, :].cpu().numpy()
        p_input['pose_pred'] = self.planner_pose_inputs
        p_input['goal'] = goal_maps  # global_goals[e]
        p_input['new_goal'] = self.l_step == args.num_local_steps - 1
        p_input['found_goal'] = found_goal
        p_input['wait'] = 0
        p_input['goal_name'] = infos['goal_name']
        if args.visualize or args.print_images:
            vlm = torch.clone(self.local_map[4:, :, :])
            vlm[-1] = 1e-5
            p_input['sem_map_pred'] = vlm.argmax(0).cpu().numpy()
            p_input['opp_score'] = maxi
            p_input['opp_cat'] = maxc
            p_input['itself'] = maxa

        self.inc_step()
        return p_input
