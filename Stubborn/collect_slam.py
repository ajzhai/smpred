import argparse
import os
import random
import habitat
import torch
import sys
import cv2
import open3d as o3d
import time
from arguments import get_args
from habitat.core.env import Env
from constants import hm3d_names
import numpy as np
import matplotlib.pyplot as plt

from agent.smp_agent import SMPAgent
from pose_est import pose_estimation

def shuffle_episodes(env, shuffle_interval):
    ranges = np.arange(0, len(env.episodes), shuffle_interval)
    np.random.shuffle(ranges)
    new_episodes = []
    for r in ranges:
        new_episodes += env.episodes[r:r + shuffle_interval]
    env.episodes = new_episodes
    
def main():

    args_2 = get_args()
    args_2.only_explore = 0  ########## whether to NOT go for goal detections 
    
    config_paths = '/challenge_objectnav2022noisy.local.rgbd.yaml'
    config = habitat.get_config(config_paths)
    
    config.defrost()
    config.SEED = 100
    # config.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = args_2.sem_gpu_id
    config.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
    config.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES = 1
    config.DATASET.SPLIT = 'val'
    config.SIMULATOR.DEPTH_SENSOR.NOISE_MODEL_KWARGS.noise_multiplier = args_2.depth_noise_mult
    config.freeze()
    print(config.DATASET.SPLIT)
    
    nav_agent = SMPAgent(args=args_2,task_config=config)
    hab_env = Env(config=config)
    
    print(len(hab_env.episodes), 'episodes in dataset')
    
    num_episodes = 500
    start = args_2.start_ep
    end = args_2.end_ep if args_2.end_ep > 0 else num_episodes
    
    save_steps = list(range(25, 525, 25))
    succs, spls, dtgs, sspls, epls = [], [], [], [], []
    
    width, height = 640, 480
    fov = 79
    xc = (width - 1.) / 2.
    zc = (height - 1.) / 2.
    f = (width / 2.) / np.tan(np.deg2rad(fov / 2.))
    intrinsic = np.identity(3)
    intrinsic[0, 0] = f
    intrinsic[1, 1] = f
    intrinsic[0, 2] = xc
    intrinsic[1, 2] = zc
    intrinsic_l = o3d.camera.PinholeCameraIntrinsic(width, height, intrinsic)


    count_episodes = 0
    while count_episodes < min(num_episodes, end):
        observations = hab_env.reset()
        print(hm3d_names[observations['objectgoal'][0]], '############' * 5)
        nav_agent.reset()
        print(hab_env._current_episode.scene_id)
        
        last_gps = observations['gps'][:2]
        last_comp = observations['compass'][0]
        last_rgb = observations['rgb'].astype(np.float32) / 255.
        last_depth = preprocess_depth(observations['depth'], 0.5, 5)
        last_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(last_rgb), o3d.geometry.Image(last_depth), depth_scale=1.0, depth_trunc=5.0)    
        pose = np.eye(4)
        if count_episodes >= start and count_episodes < end:

            step_i = 0
            seq_i = 0
            actions = []
            # full_map_seq = np.zeros((len(save_steps), 4 + args_2.num_sem_categories, nav_agent.agent_states.full_w, nav_agent.agent_states.full_h), dtype=np.uint8)
            while not hab_env.episode_over:
                sys.stdout.flush()
                
                observations['gps'][:2] += np.random.normal(scale=args_2.pose_noise_std, size=2)
                action = nav_agent.act(observations)
                observations = hab_env.step(action)
                
                curr_rgb = observations['rgb'].astype(np.float32) / 255.
                curr_depth = preprocess_depth(observations['depth'], 0.5, 5)
                curr_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(curr_rgb), o3d.geometry.Image(curr_depth), depth_scale=1.0, depth_trunc=5.0) 
                #success, trans, info = register_one_rgbd_pair(last_rgbd, curr_rgbd, action['action'], intrinsic_l, True)
                trans, fitness = grid_register(last_rgbd, curr_rgbd, action['action'], 
                                               intrinsic_l, device_id=args_2.sem_gpu_id, icp_refine=args_2.icp_refine)
                #print(success, trans)
                pose = pose @ trans
                # print(step_i, observations['gps'], '  ', observations['compass'], '    ', 
                #       observations['gps'][:2] - last_gps, '  ', observations['compass'][0] - last_comp)
                # last_gps = np.asarray(observations['gps'][:2]).copy()
                # last_comp = observations['compass'][0]
                
                
                pose[2, 3] = np.clip(pose[2, 3], a_min=-19, a_max=19)
                pose[0, 3] = np.clip(pose[0, 3], a_min=-19, a_max=19)
                print(step_i, observations['gps'][:2], [-pose[2, 3], pose[0, 3]])
                print(observations['compass'][0], heading_angle(pose[:3, :3]))
                observations['gps'][:2] = [-pose[2, 3], pose[0, 3]]
                observations['compass'][0] = heading_angle(pose[:3, :3])
                
                last_rgbd = curr_rgbd
                
                
                # if step_i in range(0, 200):
                #     cv2.imwrite('./data/tmp/ep4/rgb%d.png' % step_i, observations['rgb'][:, :, ::-1])
                # if step_i in range(0, 200):
                #     np.save('./data/tmp/ep4/depth%03d.npy' % step_i, observations['depth'])
                          
                if step_i % 100 == 50:
                    print('episode %d, step %d' % (count_episodes, step_i))
                    sys.stdout.flush()
                    break

                step_i += 1
                actions.append(action['action'])
                
            print(actions)
                
            if args_2.only_explore == 0:
                # Record final map, nav metrics, final front-view RGB
                # full_map = nav_agent.agent_states.full_map.cpu().numpy() * 255
                # full_map_seq[-1] = full_map.astype(np.uint8)
            
                metrics = hab_env.get_metrics()
                succs.append(metrics['success'])
                spls.append(metrics['spl'])
                dtgs.append(metrics['distance_to_goal'])
                sspls.append(metrics['softspl'])
                epls.append(step_i)
                stats = np.array([succs, spls, dtgs, sspls, epls])
                if 'debug' not in args_2.exp_name:
                    np.save('data/lm/logged_metrics_smp_' + args_2.exp_name + '_500.npy', stats)
                print(metrics)
                print(np.mean(stats, axis=1))
                # np.save('data/tmp/end%03d.npy' % count_episodes, observations['rgb'])
                # if args_2.print_images:
                #     cv2.imwrite('./data/tmp/rgb/rgb%d.png' % count_episodes, observations['rgb'])
                
            # np.savez_compressed('./data/saved_maps/train_rn/f%05d.npz' % count_episodes, maps=full_map_seq)

        count_episodes += 1
        
    
    
def heading_angle(R):
    axis = [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1] ]
    trace_R = np.clip(np.sum(np.diag(R)), a_min=-1, a_max=3)
    angmag = np.arccos((trace_R - 1) / 2)
    if axis[1] < 0 and angmag > 0.001:
        return -angmag
    else:
        return angmag

    
def preprocess_depth(depth, min_d, max_d):
    depth = depth[:, :] * 1

    for i in range(depth.shape[1]):
        invalid = depth[:, i] == 0.
        if np.mean(invalid) > 0.9:
            depth[:, i][invalid] = depth[:, i].max()
        else:
            depth[:, i][invalid] = np.inf #depth[:, i].max()

    mask2 = depth > 0.99
    depth[mask2] = 0.

    mask1 = depth == 0
    depth[mask1] = np.inf
    depth = min_d * 1. + depth * (max_d - min_d) * 1.
    return depth
    

def grid_register(source_rgbd_image, target_rgbd_image, action, intrinsic, device_id=0, icp_refine=False):
    start = time.time()
    
    flip = -1 if action == 3 else 1
    init_trans = np.eye(4)
    if action == 1:
        init_trans[2, 3] = -0.25
    else:
        ang = flip * 30 * np.pi / 180
        init_trans[0, 0] = np.cos(ang)
        init_trans[0, 2] = np.sin(ang)
        init_trans[2, 0] = -np.sin(ang)
        init_trans[2, 2] = np.cos(ang)
    
    try:
        source_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            source_rgbd_image, intrinsic)
        target_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            target_rgbd_image, intrinsic)
        source_pcd = o3d.t.geometry.PointCloud(
            o3d.core.Tensor(np.asarray(source_pcd.points[::4]))).cuda().voxel_down_sample(voxel_size=0.01)
        target_pcd = o3d.t.geometry.PointCloud(
            o3d.core.Tensor(np.asarray(target_pcd.points[::4]))).cuda().voxel_down_sample(voxel_size=0.01)
    except:
        return init_trans, 0
    
    best_fitness = 0
    best_trans = init_trans
    for ang_deg in (np.arange(20, 40, 1) if action in [2, 3] else [ -4, -2, 0, 2, 4]):
        for zdist in (np.arange(0.0, 0.4, 0.02) if action == 1 else [-0.04, -0.02, 0, 0.02, 0.04]):
            for xdist in (np.arange(-0.08, 0.081, 0.04) if action == 1 else [-0.04, -0.02, 0, 0.02, 0.04]):
    # for ang_deg in (np.arange(25, 35, 0.25) if action in [2, 3] else [ -2, -1, 0, 1, 2]):
    #     for zdist in (np.arange(0.0, 0.31, 0.01) if action == 1 else [0]):
    #         for xdist in (np.arange(0, 0.01, 0.01) if action == 1 else [0]):
                
                trans = np.eye(4)
                trans[2, 3] = -zdist
                trans[0, 3] = xdist
                ang = flip * ang_deg * np.pi / 180
                trans[0, 0] = np.cos(ang)
                trans[0, 2] = np.sin(ang)
                trans[2, 0] = -np.sin(ang)
                trans[2, 2] = np.cos(ang)
                fitness = o3d.t.pipelines.registration.evaluate_registration(source_pcd, 
                    target_pcd, 0.03, transformation=trans).fitness
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_trans = trans
                    
    if icp_refine:
        icp_result = o3d.t.pipelines.registration.icp(source_pcd, target_pcd, 0.03, best_trans)
        best_trans = icp_result.transformation.cpu().numpy()
    if np.sum(np.isnan(best_trans)) > 0:
        best_trans = init_trans
    return best_trans, best_fitness

def register_one_rgbd_pair(source_rgbd_image, target_rgbd_image, action, intrinsic,
                           with_opencv):

    option = o3d.pipelines.odometry.OdometryOption()
    option.depth_diff_max = 0.05
    option.depth_max = 5.
    start = time.time()
    if with_opencv:
        # success_5pt, odo_init = pose_estimation(source_rgbd_image,
        #                                         target_rgbd_image,
        #                                         intrinsic, False)
        odo_init = np.eye(4)
        success_5pt = True
        ang = np.pi / 6
        if action == 1:
            success_5pt, odo_init_orb = pose_estimation(source_rgbd_image,
                                                target_rgbd_image,
                                                intrinsic, False)
            if abs(odo_init_orb[2, 3] - (-0.25)) > 0.5 or np.isnan(odo_init_orb[2, 3]):
                odo_init[2, 3] = -0.25
            else:
                odo_init[2, 3] = odo_init_orb[2, 3]
            # odo_init[2, 3] += np.random.normal(0, 0.05)
            # odo_init[0, 3] += np.random.normal(0, 0.05)
        elif action == 2:
            odo_init[0, 0] = np.cos(ang)
            odo_init[0, 2] = np.sin(ang)
            odo_init[2, 0] = -np.sin(ang)
            odo_init[2, 2] = np.cos(ang)
            # odo_init[2, 3] += np.random.normal(0, 0.05)
            # odo_init[0, 3] += np.random.normal(0, 0.05)
        elif action == 3:
            ang = -ang
            odo_init[0, 0] = np.cos(ang)
            odo_init[0, 2] = np.sin(ang)
            odo_init[2, 0] = -np.sin(ang)
            odo_init[2, 2] = np.cos(ang)
            # odo_init[2, 3] += np.random.normal(0, 0.05)
            # odo_init[0, 3] += np.random.normal(0, 0.05)
        #print(i + 1, odo_init[[0, 2], 3], myangle(odo_init[:3, :3]))
        #print(time.time() - start)
        #print(odo_init)
        [success, trans, info] = o3d.pipelines.odometry.compute_rgbd_odometry(
            source_rgbd_image, target_rgbd_image, intrinsic, odo_init,
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
            option)
        #print(time.time() - start)
        #print(i + 1, trans[[0, 2], 3], myangle(trans[:3, :3]))
        

        trans[1, :3] = 0
        trans[:3, 1] = 0
        trans[1, 1] = 1
        trans[:2, 3] = 0
        
        if action == 1:
            if abs(trans[2, 3] - odo_init[2, 3]) > 0.5  or np.sum(np.isnan(odo_init_orb[2, 3])) > 0:
                trans = odo_init
        elif action in [2, 3]:
            if abs(heading_angle(trans[:3, :3]) - ang) > 0.2:
                trans = odo_init

        return [success, trans, info]
    return [False, np.identity(4), np.identity(6)]


if __name__ == "__main__":
    main()
