import argparse
import os
import random
import habitat
import torch
import sys
import cv2
import pickle
from arguments import get_args
from habitat.core.env import Env
from constants import hm3d_names, mpcat40_labels, raw_name_to_mpcat40
import numpy as np
import matplotlib.pyplot as plt

from agent.smp_agent import SMPAgent

TO_COLLECT = ['chair', 'sofa', 'plant', 'bed', 'toilet', 'tv_monitor', 'sink', 
              'fireplace', 'cabinet', 'bathtub', 'mirror', 'cushion', 'chest_of_drawers']


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
    config.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
    config.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES = 10
    config.SIMULATOR.SCENE_DATASET = 'habitat-challenge-data/data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json'
    config.DATASET.SPLIT = 'val'
    config.freeze()
    print(config.DATASET.SPLIT)
    
    nav_agent = SMPAgent(args=args_2,task_config=config)
    hab_env = Env(config=config)
    
    print(len(hab_env.episodes), 'episodes in dataset')
    
    num_episodes = 200
    start = args_2.start_ep
    end = args_2.end_ep if args_2.end_ep > 0 else num_episodes

    scene_visits = {}
    
    count_episodes = 0
    saved = 0
    while count_episodes < num_episodes:
        observations = hab_env.reset()
        nav_agent.reset()
        
        annots = hab_env.sim.semantic_annotations()
        instance_id_to_label_id = {int(obj.id.split("_")[-1]): obj.category.index() for obj in annots.objects}
        # for obj in annots.objects:
        #     print(obj.category.index(), obj.category.name())
                
        instance_id_to_cat_name = {int(obj.id.split("_")[-1]): obj.category.name() for obj in annots.objects}
        scene = hab_env.current_episode.scene_id
        if len(observations.keys()) < 6: 
            scene_visits[scene] = 0
            continue
        else:
            sys.stdout.flush()
            if scene not in scene_visits:
                scene_visits[scene] = 1
            elif scene_visits[scene] == config.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES:
                continue
            else:
                scene_visits[scene] += 1
            
        if count_episodes >= start and count_episodes < end:

            step_i = 0
            
            while step_i < 100 and not hab_env.episode_over:
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

                # Record front-view RGB and semseg
                sem_obs = observations['semantic']
                o_lst = []
                for o_i, o_id in enumerate(np.unique(sem_obs)):
                    o_cat = instance_id_to_cat_name[o_id]
                    if o_cat in raw_name_to_mpcat40:
                        o_cat = raw_name_to_mpcat40[o_cat]
                    else:
                        continue
                    if o_cat in TO_COLLECT:
                        o_dict = {}
                        #o_dict['cat_id'] = o_cat
                        o_dict['cat'] = o_cat #mpcat40_labels[o_cat]
                        o_dict['idxs'] = np.where(sem_obs == o_id)
                        o_lst.append(o_dict)
                
                pickle.dump(o_lst, open('data/seg/%s/sem/%03d_%03d.pkl' % (config.DATASET.SPLIT, count_episodes, step_i), 'wb'))
                
                cv2.imwrite('data/seg/%s/rgb/%03d_%03d.png' % (config.DATASET.SPLIT, count_episodes, step_i), 
                            observations['rgb'][:, :, ::-1].astype(np.uint8))
                # if args_2.print_images:
                #     cv2.imwrite('./data/tmp/rgb/rgb%d.png' % count_episodes, observations['rgb'])
                saved += 1
                
                step_i += 1
                

        count_episodes += 1
        
    print(saved, 'total img')
    print(len(scene_visits), 'total scene')
    

if __name__ == "__main__":
    main()
