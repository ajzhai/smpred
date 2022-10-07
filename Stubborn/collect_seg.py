import argparse
import os
import random
import habitat
import torch
import sys
import cv2
from arguments import get_args
from habitat.core.env import Env
from constants import hm3d_names
import numpy as np
import matplotlib.pyplot as plt

from agent.smp_agent import SMPAgent

def shuffle_episodes(env, shuffle_interval):
    ranges = np.arange(0, len(env.episodes), shuffle_interval)
    np.random.shuffle(ranges)
    new_episodes = []
    for r in ranges:
        new_episodes += env.episodes[r:r + shuffle_interval]
    env.episodes = new_episodes
    
def main():

    args_2 = get_args()
    args_2.only_explore = 1  ########## whether to NOT go for goal detections 
    
    config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
    config = habitat.get_config(config_paths)
    config.defrost()
    config.SEED = 100
    # config.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = args_2.sem_gpu_id
    config.SIMULATOR.AGENT_0.SENSORS =  ['RGB_SENSOR', 'DEPTH_SENSOR', 'SEMANTIC_SENSOR']
    config.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES = 1
    config.SIMULATOR.SCENE_DATASET = 'habitat-challenge-data/data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json'
    config.DATASET.SPLIT = 'val'
    config.freeze()
    print(config.DATASET.SPLIT)
    
    nav_agent = SMPAgent(args=args_2,task_config=config)
    hab_env = Env(config=config)
    
    print(len(hab_env.episodes), 'episodes in dataset')
    
    num_episodes = 20
    start = args_2.start_ep
    end = args_2.end_ep if args_2.end_ep > 0 else num_episodes

    scene_visits = {}
    
    count_episodes = 0
    while count_episodes < num_episodes:
        observations = hab_env.reset()
        nav_agent.reset()
        
        scene = hab_env.current_episode.scene_id
        if len(observations.keys()) < 6: 
            scene_visits[scene] = 0
            continue
        else:
            print(observations.keys())
            sys.stdout.flush()
            if scene not in scene_visits:
                scene_visits[scene] = 1
            elif scene_visits[scene] == 1: ##
                continue
            else:
                scene_visits[scene] += 1
        
        if len(scene_visits) == 10:
            print(count_episodes)
            exit(0)
            
        if count_episodes >= start and count_episodes < end:

            step_i = 0
            seq_i = 0
            while step_i < 2 and not hab_env.episode_over:
                sys.stdout.flush()
                action = nav_agent.act(observations)
                observations = hab_env.step(action)
                # cv2.imwrite('./data/tmp/rgb/rgb%d.png' % step_i, observations['rgb'][:, :, ::-1])
                # if step_i in range(21, 32):
                #     #print(step_i, observations['gps'], observations['compass'])
                #     np.save('data/tmp/rgb%03d.npy' % step_i, observations['rgb'])
                          
                if step_i % 100 == 0:
                    print('\n\n episode %d, step %d' % (count_episodes, step_i))
                    sys.stdout.flush()

                step_i += 1
                    

                # Record final front-view RGB and semseg
                np.save('data/tmp/example_sem.npy', observations['semantic'])
                # if args_2.print_images:
                #     cv2.imwrite('./data/tmp/rgb/rgb%d.png' % count_episodes, observations['rgb'])
                

        count_episodes += 1
        
    

if __name__ == "__main__":
    main()
