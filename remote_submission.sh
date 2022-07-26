#!/usr/bin/env bash

# python Stubborn/collect.py -v 0 --dump_location ./data/tmp --print_images 1 --timestep_limit 1000 --evaluation $AGENT_EVALUATION_TYPE $@

python Stubborn/collect.py --sem_gpu_id 2 --start_ep 0 --end_ep 2 --timestep_limit 1000 --evaluation $AGENT_EVALUATION_TYPE $@  &
python Stubborn/collect.py --sem_gpu_id 3 --start_ep 2 --end_ep 4 --timestep_limit 1000 --evaluation $AGENT_EVALUATION_TYPE $@  &
wait
