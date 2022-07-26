import argparse
import os
import random
import habitat
import torch
import sys
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
    config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = args_2.sem_gpu_id
    config.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES = 50
    config.freeze()
    print(config.DATASET.SPLIT)
    
    nav_agent = SMPAgent(args=args_2,task_config=config)
    hab_env = Env(config=config)
    
    print(len(hab_env.episodes), 'episodes in dataset')
    
    num_episodes = 200
    start = args_2.start_ep
    end = args_2.end_ep if args_2.end_ep > 0 else num_episodes
    
    save_steps = list(range(25, 525, 25))
    succs, spls, dtgs, epls = [], [], [], []
    
    count_episodes = 0
    while count_episodes < num_episodes:
        observations = hab_env.reset()
        # print(hm3d_names[observations['objectgoal'][0]], '############' * 5)
        nav_agent.reset()
        print(hab_env._current_episode.scene_id)
        
        if count_episodes >= start and count_episodes < end:

            step_i = 0
            seq_i = 0
            full_map_seq = np.zeros((len(save_steps), 4 + args_2.num_sem_categories, nav_agent.agent_states.full_w, nav_agent.agent_states.full_h), dtype=np.uint8)
            while not hab_env.episode_over:
                action = nav_agent.act(observations)
                observations = hab_env.step(action)

                if step_i % 100 == 0:
                    print('episode %d, step %d' % (count_episodes, step_i))
                    sys.stdout.flush()

                step_i += 1
                if step_i in save_steps:
                    full_map = nav_agent.agent_states.full_map.cpu().numpy() * 255
                    full_map_seq[seq_i] = full_map.astype(np.uint8)
                    seq_i += 1
                    
        
            if args_2.only_explore == 0:
                # Record final map, nav metrics, final front-view RGB
                full_map = nav_agent.agent_states.full_map.cpu().numpy() * 255
                full_map_seq[-1] = full_map.astype(np.uint8)
            
                metrics = hab_env.get_metrics()
                succs.append(metrics['success'])
                spls.append(metrics['spl'])
                dtgs.append(metrics['distance_to_goal'])
                epls.append(step_i)

                plt.imshow(observations['rgb'])
                plt.savefig('./data/tmp/end%d.png' % count_episodes)
                plt.close()
                
            np.savez_compressed('./data/saved_maps/temp/f%05d.npz' % count_episodes, maps=full_map_seq)

        count_episodes += 1
        
    if args_2.only_explore == 0:
        stats = np.array([succs, spls, dtgs, epls])
        np.save('data/tmp/logged_metrics.npy', stats)
    

if __name__ == "__main__":
    main()
