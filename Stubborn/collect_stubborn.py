import argparse
import os
import random
import habitat
import torch
import sys
import cv2
from stubborn_arguments import get_args
from habitat.core.env import Env
from constants import hm3d_names, habitat_labels_r
import numpy as np
import matplotlib.pyplot as plt

from agent.stubborn_agent import StubbornAgent

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
    
    args_2.num_sem_categories = 23
    
    config_paths = '/challenge_objectnav2021.local.rgbd.yaml'  # 2021 MP3D
    config = habitat.get_config(config_paths)
    config.defrost()
    config.SEED = 100
    # config.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = args_2.sem_gpu_id
    config.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
    config.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES = -1
    config.DATASET.SPLIT = 'val'
    config.freeze()
    print(config.DATASET.SPLIT)
    
    nav_agent = StubbornAgent(args=args_2,task_config=config)
    hab_env = Env(config=config)
    
    print(len(hab_env.episodes), 'episodes in dataset')
    
    num_episodes = len(hab_env.episodes)
    start = args_2.start_ep
    end = args_2.end_ep if args_2.end_ep > 0 else num_episodes
    
    save_steps = list(range(25, 525, 25))
    succs, spls, dtgs, sspls, epls = [], [], [], [], []
    
    count_episodes = 0
    while count_episodes < min(num_episodes, end):
        observations = hab_env.reset()
        print(habitat_labels_r[observations['objectgoal'][0] + 1], '############' * 5)
        nav_agent.reset()
        print(hab_env._current_episode.scene_id)
        
        if count_episodes >= start and count_episodes < end:

            step_i = 0
            seq_i = 0
            while not hab_env.episode_over:
                sys.stdout.flush()
                action = nav_agent.act(observations)
                observations = hab_env.step(action)
                
                if step_i % 100 == 0:
                    print('episode %d, step %d' % (count_episodes, step_i))
                    sys.stdout.flush()

                step_i += 1
                # if step_i in save_steps:
                #     full_map = nav_agent.agent_states.full_map.cpu().numpy() * 255
                #     full_map_seq[seq_i] = full_map.astype(np.uint8)
                #     seq_i += 1
                    
        
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
                # np.save('data/tmp/logged_metrics_smp_a%04d.npy' % args_2.alpha, stats)
                if args_2.exp_name != 'debug':
                    np.save('data/lm/logged_metrics_stb_' + args_2.exp_name + '_%dmp3d.npy' % num_episodes, stats)
                print(metrics)
                # np.save('data/tmp/end%03d.npy' % count_episodes, observations['rgb'])
                # if args_2.print_images:
                #     cv2.imwrite('./data/tmp/rgb/rgb%d.png' % count_episodes, observations['rgb'])
                
            # np.savez_compressed('./data/saved_maps/train_rn/f%05d.npz' % count_episodes, maps=full_map_seq)

        count_episodes += 1
        
    

if __name__ == "__main__":
    main()
