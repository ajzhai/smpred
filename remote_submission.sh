#!/usr/bin/env bash

#python Stubborn/collect.py -v 0 --dump_location ./data/tmp --exp_name debug --print_images 1 --sem_pred_prob_thr 0.9 --smp_step 20 --switch_step 19 --alpha 200 --col_rad 4 --start_ep 2 --sf_thr '-2' --evaluation $AGENT_EVALUATION_TYPE $@ 
#python Stubborn/collect.py --dump_location ./data/tmp --exp_name debug --print_images 1 --switch_step 501 --map_resolution 2 --evaluation $AGENT_EVALUATION_TYPE $@  # Stubborn (rednet)

python Stubborn/collect.py --sem_gpu_id 1 --exp_name a400_sms20 --print_images 0 --sem_pred_prob_thr 0.9 --smp_step 20 --switch_step 19 --alpha 400 --col_rad 4 --start_ep 0 --evaluation $AGENT_EVALUATION_TYPE $@ &
python Stubborn/collect.py --sem_gpu_id 1 --exp_name a200_sms20 --print_images 0 --sem_pred_prob_thr 0.9 --smp_step 20 --switch_step 19 --alpha 200 --col_rad 4 --start_ep 0 --evaluation $AGENT_EVALUATION_TYPE $@ &
python Stubborn/collect.py --sem_gpu_id 3 --exp_name a400_sms10 --print_images 0 --sem_pred_prob_thr 0.9 --smp_step 10 --switch_step 19 --alpha 400 --col_rad 4 --start_ep 0 --evaluation $AGENT_EVALUATION_TYPE $@ &
python Stubborn/collect.py --sem_gpu_id 3 --exp_name a200_sms10 --print_images 0 --sem_pred_prob_thr 0.9 --smp_step 10 --switch_step 19 --alpha 200 --col_rad 4 --start_ep 0 --evaluation $AGENT_EVALUATION_TYPE $@ &

# python Stubborn/collect.py --sem_gpu_id 8 --start_ep 0 --end_ep 2000 --timestep_limit 1000 --evaluation $AGENT_EVALUATION_TYPE $@  &
# python Stubborn/collect.py --sem_gpu_id 8 --start_ep 2000 --end_ep 4000 --timestep_limit 1000 --evaluation $AGENT_EVALUATION_TYPE $@  &
# python Stubborn/collect.py --sem_gpu_id 8 --start_ep 4000 --end_ep 6000 --timestep_limit 1000 --evaluation $AGENT_EVALUATION_TYPE $@  &
# python Stubborn/collect.py --sem_gpu_id 8 --start_ep 6000 --end_ep 8000 --timestep_limit 1000 --evaluation $AGENT_EVALUATION_TYPE $@  &
# python Stubborn/collect.py --sem_gpu_id 5 --start_ep 8000 --end_ep 10000 --timestep_limit 1000 --evaluation $AGENT_EVALUATION_TYPE $@  &
# python Stubborn/collect.py --sem_gpu_id 5 --start_ep 10000 --end_ep 12000 --timestep_limit 1000 --evaluation $AGENT_EVALUATION_TYPE $@  &
# python Stubborn/collect.py --sem_gpu_id 5 --start_ep 12000 --end_ep 14000 --timestep_limit 1000 --evaluation $AGENT_EVALUATION_TYPE $@  &
# python Stubborn/collect.py --sem_gpu_id 5 --start_ep 14000 --end_ep 16000 --timestep_limit 1000 --evaluation $AGENT_EVALUATION_TYPE $@  &
# python Stubborn/collect.py --sem_gpu_id 6 --start_ep 16000 --end_ep 18000 --timestep_limit 1000 --evaluation $AGENT_EVALUATION_TYPE $@  &
# python Stubborn/collect.py --sem_gpu_id 6 --start_ep 18000 --end_ep 20000 --timestep_limit 1000 --evaluation $AGENT_EVALUATION_TYPE $@  &
# python Stubborn/collect.py --sem_gpu_id 6 --start_ep 20000 --end_ep 22000 --timestep_limit 1000 --evaluation $AGENT_EVALUATION_TYPE $@  &
# python Stubborn/collect.py --sem_gpu_id 6 --start_ep 22000 --end_ep 24000 --timestep_limit 1000 --evaluation $AGENT_EVALUATION_TYPE $@  &
# python Stubborn/collect.py --sem_gpu_id 7 --start_ep 24000 --end_ep 26000 --timestep_limit 1000 --evaluation $AGENT_EVALUATION_TYPE $@  &
# python Stubborn/collect.py --sem_gpu_id 7 --start_ep 26000 --end_ep 28000 --timestep_limit 1000 --evaluation $AGENT_EVALUATION_TYPE $@  &
# python Stubborn/collect.py --sem_gpu_id 7 --start_ep 28000 --end_ep 30000 --timestep_limit 1000 --evaluation $AGENT_EVALUATION_TYPE $@  &
# python Stubborn/collect.py --sem_gpu_id 7 --start_ep 30000 --end_ep 32000 --timestep_limit 1000 --evaluation $AGENT_EVALUATION_TYPE $@  &
# python Stubborn/collect.py --sem_gpu_id 8 --start_ep 32000 --end_ep 34000 --timestep_limit 1000 --evaluation $AGENT_EVALUATION_TYPE $@  &
# python Stubborn/collect.py --sem_gpu_id 8 --start_ep 34000 --end_ep 36000 --timestep_limit 1000 --evaluation $AGENT_EVALUATION_TYPE $@  &
# python Stubborn/collect.py --sem_gpu_id 8 --start_ep 36000 --end_ep 38000 --timestep_limit 1000 --evaluation $AGENT_EVALUATION_TYPE $@  &
# python Stubborn/collect.py --sem_gpu_id 8 --start_ep 38000 --end_ep 40000 --timestep_limit 1000 --evaluation $AGENT_EVALUATION_TYPE $@  &
wait
