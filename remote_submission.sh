#!/usr/bin/env bash

#python Stubborn/collect_mp3d.py -v 0 --dump_location ./data/tmp --exp_name mp3d_debug --print_images 1 --evaluation $AGENT_EVALUATION_TYPE $@ 
#python Stubborn/collect.py --dump_location ./data/tmp --exp_name debug --print_images 1 --switch_step 501 --map_resolution 2 --evaluation $AGENT_EVALUATION_TYPE $@  # Stubborn (rednet)

python Stubborn/collect_mp3d.py -v 0 --sem_gpu_id 0 --exp_name mp3d_a200 --print_images 0 --alpha 200 --evaluation $AGENT_EVALUATION_TYPE $@ 
python Stubborn/collect_mp3d.py -v 0 --sem_gpu_id 0 --exp_name mp3d_a250 --print_images 0 --alpha 250 --evaluation $AGENT_EVALUATION_TYPE $@ 
python Stubborn/collect_mp3d.py -v 0 --sem_gpu_id 0 --exp_name mp3d_a300 --print_images 0 --alpha 300 --evaluation $AGENT_EVALUATION_TYPE $@ 
# python Stubborn/collect.py --sem_gpu_id 2 --exp_name gthr_9825_highsofa --print_images 0 --goal_thr 0.9825 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect.py --sem_gpu_id 2 --exp_name gthr_9875_highsofa --print_images 0 --goal_thr 0.9875 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect.py --sem_gpu_id 2 --exp_name gthr_985_e1200 --print_images 0 --goal_thr 0.985 --escape 1200 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect.py --sem_gpu_id 2 --exp_name gthr_985_ge2 --print_images 0 --goal_erode 2 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect.py --sem_gpu_id 2 --exp_name gthr_985_ge4 --print_images 0 --goal_erode 4 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect.py --sem_gpu_id 4 --exp_name thr_90_95_a200 --print_images 0 --tv_thr 0.9 --sem_pred_prob_thr 0.95 --alpha 200 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect.py --sem_gpu_id 4 --exp_name thr_90_95_a300 --print_images 0 --tv_thr 0.9 --sem_pred_prob_thr 0.95 --alpha 300 --evaluation $AGENT_EVALUATION_TYPE $@ &

# python Stubborn/collect.py --sem_gpu_id 2 --exp_name a500_new --print_images 0 --sem_pred_prob_thr 0.9 --smp_step 10 --switch_step 19 --alpha 500 --col_rad 4 --start_ep 0 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect.py --sem_gpu_id 2 --exp_name a400_new --print_images 0 --sem_pred_prob_thr 0.9 --smp_step 10 --switch_step 19 --alpha 400 --col_rad 4 --start_ep 0 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect.py --sem_gpu_id 2 --exp_name a300_new --print_images 0 --sem_pred_prob_thr 0.9 --smp_step 10 --switch_step 19 --alpha 300 --col_rad 4 --start_ep 0 --evaluation $AGENT_EVALUATION_TYPE $@ &

# python Stubborn/collect.py --sem_gpu_id 0 --exp_name a400_sms10_conf80 --print_images 0 --sem_pred_prob_thr 0.8 --smp_step 10 --switch_step 19 --alpha 400 --col_rad 4 --start_ep 0 --escape 0 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect.py --sem_gpu_id 1 --exp_name a400_sms10_escape --print_images 0 --sem_pred_prob_thr 0.9 --smp_step 10 --switch_step 19 --alpha 400 --col_rad 4 --start_ep 0 --escape 1 --evaluation $AGENT_EVALUATION_TYPE $@ &
#python Stubborn/collect.py --sem_gpu_id 0 --exp_name rn_a400_sms10 --print_images 1 --sem_pred_prob_thr 0.9 --smp_step 10 --switch_step 19 --alpha 400 --col_rad 3.5 --start_ep 0 --escape 0 --num_sem_categories 23 --global_downscaling 3 --map_size_cm 4800 --evaluation $AGENT_EVALUATION_TYPE $@ &

# python Stubborn/collect_maps_mp3d.py --sem_gpu_id 1 --start_ep 0 --end_ep 110 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_maps_mp3d.py --sem_gpu_id 1 --start_ep 110 --end_ep 220 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_maps_mp3d.py --sem_gpu_id 1 --start_ep 220 --end_ep 330 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_maps_mp3d.py --sem_gpu_id 1 --start_ep 330 --end_ep 440 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_maps_mp3d.py --sem_gpu_id 1 --start_ep 440 --end_ep 550 --evaluation $AGENT_EVALUATION_TYPE $@ &

# python Stubborn/collect_maps.py --sem_gpu_id 0 --start_ep 0 --end_ep 125 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_maps.py --sem_gpu_id 0 --start_ep 125 --end_ep 250 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_maps.py --sem_gpu_id 0 --start_ep 250 --end_ep 375 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_maps.py --sem_gpu_id 0 --start_ep 375 --end_ep 500 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_maps.py --sem_gpu_id 0 --start_ep 500 --end_ep 625 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_maps.py --sem_gpu_id 0 --start_ep 625 --end_ep 750 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_maps.py --sem_gpu_id 0 --start_ep 750 --end_ep 875 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_maps.py --sem_gpu_id 0 --start_ep 875 --end_ep 1000 --evaluation $AGENT_EVALUATION_TYPE $@ &

# python Stubborn/collect_maps_800.py --sem_gpu_id 4 --start_ep 0000 --end_ep 1000 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_maps_800.py --sem_gpu_id 4 --start_ep 1000 --end_ep 2000 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_maps_800.py --sem_gpu_id 4 --start_ep 2000 --end_ep 3000 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_maps_800.py --sem_gpu_id 4 --start_ep 3000 --end_ep 4000 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_maps_800.py --sem_gpu_id 4 --start_ep 4000 --end_ep 5000 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_maps_800.py --sem_gpu_id 4 --start_ep 5000 --end_ep 6000 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_maps_800.py --sem_gpu_id 4 --start_ep 6000 --end_ep 7000 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_maps_800.py --sem_gpu_id 4 --start_ep 7000 --end_ep 8000 --evaluation $AGENT_EVALUATION_TYPE $@ &

wait
