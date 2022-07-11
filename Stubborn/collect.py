import argparse
import os
import random
import habitat
import torch
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
    nav_agent = SMPAgent(args=args_2,task_config=config)
    hab_env = Env(config=config)
    hab_env.dataset.max_scene_repetition_episodes = 2
    
    num_episodes = 4
    #print(len(hab_env.episodes))
    
    count_episodes = 0
    while count_episodes < num_episodes:
        observations = hab_env.reset()
        nav_agent.reset()

        step_i = 0
        seq_i = 0
        full_map_seq = np.zeros((5, args.num_sem_categories, nav_agent.agent_states.full_w, nav_agent.agent_states.full_h))
        while not hab_env.episode_over:
            action = nav_agent.act(observations)
            observations = hab_env.step(action)
            
            step_i += 1
            if step_i in [25, 50, 75, 100, 500]:
                full_map_seq[seq_i] = nav_agent.agent_states.full_map[0].cpu().numpy()
                seq_i += 1
        np.save('./saved_maps/f%05d.npy' % count_episodes, full_map_seq)

        count_episodes += 1

    

if __name__ == "__main__":
    main()
