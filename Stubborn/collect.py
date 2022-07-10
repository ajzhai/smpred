import argparse
import os
import random
import habitat
import torch
from arguments import get_args
from habitat.core.env import Env
import numpy as np

from agent.stubborn_agent import StubbornAgent

def main():

    args_2 = get_args()
    args_2.sem_gpu_id = 0
    config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
    config = habitat.get_config(config_paths)
    nav_agent = StubbornAgent(args=args_2,task_config=config)
    hab_env = Env(config=config)
    
    num_episodes = len(hab_env.episodes)
    print(num_episodes)
    exit(0)
    count_episodes = 0
    while count_episodes < num_episodes:
        observations = hab_env.reset()
        nav_agent.reset()

        while not hab_env.episode_over:
            action = nav_agent.act(observations)
            observations = hab_env.step(action)

        count_episodes += 1

    

if __name__ == "__main__":
    main()
