import argparse
import os
import random
import habitat
import torch
from arguments import get_args
import numpy as np
import agent.utils.pose as pu
from constants import coco_categories, hab2coco, hab2name, habitat_labels_r, fourty221, fourty221_ori, habitat_goal_label_to_similar_coco, hm3d_names, hm3d_to_coco, hm3d_to_21
import copy
from agent.smp_state import Agent_State
from agent.smp_helper import Agent_Helper



class SMPAgent(habitat.Agent):
    def __init__(self,args,task_config: habitat.Config):
        self._POSSIBLE_ACTIONS = task_config.TASK.POSSIBLE_ACTIONS
        self.agent_states = Agent_State(args)
        self.agent_helper = Agent_Helper(args,self.agent_states)
        self.agent_states.helper = self.agent_helper
        self.last_sim_location = None
        self.device = args.device
        self.first_obs = True
        self.valid_goals = 0
        self.total_episodes = 0
        self.args = args
        self.timestep = 0
        self.low_score_threshold = 0.5
        self.high_score_threshold = 0.9
        # towel tv shower gym clothes
        # use a lower confidence score threshold for those categories
        self.low_score_categories = {13,14,15,19,21}
    def reset(self):
        self.agent_helper.reset()
        self.agent_states.reset()
        self.last_sim_location = None
        self.first_obs = True
        self.step = 0
        self.timestep = 0
        self.total_episodes += 1

    def act(self, observations):
        self.timestep += 1
        # if passed the step limit and we haven't found the goal, stop.
        # if self.timestep > self.args.timestep_limit and self.agent_states.found_goal == False:
        #     return {'action': 0}
        if self.timestep > self.args.timestep_limit:
            return {'action': 0}
        #get first preprocess
        goal = observations['objectgoal'][0]
        
        if goal in self.low_score_categories:
            self.agent_states.score_threshold = self.low_score_threshold

        info = self.get_info(observations)
        if self.args.use_gt_seg:
            info['goalseg'] = observations['goalseg']
            
        # get second preprocess
        if self.args.num_sem_categories <= 16:
            info['goal_name'] = hm3d_names[goal]
            goal = hm3d_to_coco[goal]
        elif self.args.num_sem_categories == 23:
            goal += 1
            info['goal_name'] = habitat_labels_r[goal]
        else:
            assert False
            
        self.agent_helper.set_goal_cat(goal)
        obs, info = self.agent_helper.preprocess_inputs(observations['rgb'],observations['depth'],info)
        info['goal_cat_id'] = goal
        
        obs = obs[np.newaxis,:,:,:]
        # now ready to be passed to agent states
        obs = torch.from_numpy(obs).float().to(self.device)
        if self.first_obs:
            self.agent_states.init_with_obs(obs,info)
            self.first_obs = False


        planner_inputs = self.agent_states.upd_agent_state(obs,info)
        # now get action
        action = self.agent_helper.plan_act_and_preprocess(planner_inputs)
        # For data collection purpose, collect data to train the object detection module
        if self.args.no_stop == 1 and action['action'] == 0:
            self.agent_states.clear_goal_and_set_gt_map(planner_inputs['goal'])
            return {'action':1}
        return action

    def get_info(self, obs):

        info = {}
        dx, dy, do = self.get_pose_change(obs)
        info['sensor_pose'] = [dx, dy, do]
        
        # set goal
        return info

    def get_sim_location(self,obs):
        """Returns x, y, o pose of the agent in the Habitat simulator."""

        nap2 = obs['gps'][0]
        nap0 = -obs['gps'][1]
        x = nap2
        y = nap0
        o = obs['compass']
        if o > np.pi:
            o -= 2 * np.pi
        return x, y, o

    def get_pose_change(self,obs):
        curr_sim_pose = self.get_sim_location(obs)
        if self.last_sim_location is not None:
            dx, dy, do = pu.get_rel_pose_change(
                curr_sim_pose, self.last_sim_location)
            dx,dy,do = dx[0],dy[0],do[0]
        else:
            dx, dy, do = 0,0,0
        self.last_sim_location = curr_sim_pose
        return dx, dy, do





