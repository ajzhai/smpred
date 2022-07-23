import argparse
import os
import random
import habitat
import torch
import sys
from arguments import get_args
from habitat.core.env import Env
import numpy as np

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
    args_2.sem_gpu_id = 0
    args_2.only_explore = 1
    config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
    config = habitat.get_config(config_paths)
    print(config.DATASET.SPLIT)
    
    nav_agent = SMPAgent(args=args_2,task_config=config)
    hab_env = Env(config=config)
    
    print(len(hab_env.episodes))
    
    num_episodes = 0
    #print(len(hab_env.episodes))
    
    count_episodes = 0
    while count_episodes < num_episodes:
        observations = hab_env.reset()
        nav_agent.reset()
        print(hab_env._current_episode.scene_id)
        
        if count_episodes > 2776:

            step_i = 0
            seq_i = 0
            full_map_seq = np.zeros((5, 4 + args_2.num_sem_categories, nav_agent.agent_states.full_w, nav_agent.agent_states.full_h), dtype=np.uint8)
            while not hab_env.episode_over:
                action = nav_agent.act(observations)
                observations = hab_env.step(action)

                if step_i % 100 == 0:
                    print('episode %d, step %d' % (count_episodes, step_i))
                    sys.stdout.flush()

                step_i += 1
                if step_i in [25, 50, 100, 200, 500]:
                    full_map = nav_agent.agent_states.full_map.cpu().numpy() * 255
                    full_map_seq[seq_i] = full_map.astype(np.uint8)
                    seq_i += 1
            np.savez_compressed('./data/saved_maps/train/f%05d.npz' % count_episodes, maps=full_map_seq)

        count_episodes += 1

    

if __name__ == "__main__":
    main()
