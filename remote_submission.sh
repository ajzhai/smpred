#!/usr/bin/env bash

#python Stubborn/collect_maps.py -v 0 --dump_location ./data/tmp --exp_name debug --print_images 0 --switch_step 501 --evaluation $AGENT_EVALUATION_TYPE $@ 
#python Stubborn/collect.py --dump_location ./data/tmp --exp_name debug --print_images 1 --switch_step 501 --map_resolution 2 --evaluation $AGENT_EVALUATION_TYPE $@  # Stubborn (rednet)


# python Stubborn/collect.py --sem_gpu_id 0 --exp_name tf_eoo_st20_es7 --print_images 0 --escape 700 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect.py --sem_gpu_id 0 --exp_name tf_eoo_st20_es9 --print_images 0 --escape 900 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect.py --sem_gpu_id 1 --exp_name tf_eoo_st15_es8 --print_images 0 --stair_thr 0.15 --escape 800 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect.py --sem_gpu_id 1 --exp_name tf_eoo_st15_es16 --print_images 0 --stair_thr 0.15 --escape 1600 --evaluation $AGENT_EVALUATION_TYPE $@ &
#python Stubborn/collect.py --sem_gpu_id 4 --exp_name r101_lsj90 --print_images 0 --sem_pred_prob_thr 0.9 --tv_thr 0.9 --alpha 250 --evaluation $AGENT_EVALUATION_TYPE $@ &
#python Stubborn/collect.py --sem_gpu_id 5 --exp_name r50_bi_tta --print_images 0 --sem_pred_prob_thr 0.9 --tv_thr 0.9 --alpha 250 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect.py --sem_gpu_id 0 --exp_name cr45 --print_images 0 --sem_pred_prob_thr 0.9 --smp_step 10 --switch_step 19 --alpha 400 --col_rad 4.5 --start_ep 0  --evaluation $AGENT_EVALUATION_TYPE $@ &

# python Stubborn/collect.py --sem_gpu_id 2 --exp_name a500_new --print_images 0 --sem_pred_prob_thr 0.9 --smp_step 10 --switch_step 19 --alpha 500 --col_rad 4 --start_ep 0 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect.py --sem_gpu_id 2 --exp_name a400_new --print_images 0 --sem_pred_prob_thr 0.9 --smp_step 10 --switch_step 19 --alpha 400 --col_rad 4 --start_ep 0 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect.py --sem_gpu_id 2 --exp_name a300_new --print_images 0 --sem_pred_prob_thr 0.9 --smp_step 10 --switch_step 19 --alpha 300 --col_rad 4 --start_ep 0 --evaluation $AGENT_EVALUATION_TYPE $@ &

# python Stubborn/collect.py --sem_gpu_id 0 --exp_name a400_sms10_conf80 --print_images 0 --sem_pred_prob_thr 0.8 --smp_step 10 --switch_step 19 --alpha 400 --col_rad 4 --start_ep 0 --escape 0 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect.py --sem_gpu_id 1 --exp_name a400_sms10_escape --print_images 0 --sem_pred_prob_thr 0.9 --smp_step 10 --switch_step 19 --alpha 400 --col_rad 4 --start_ep 0 --escape 1 --evaluation $AGENT_EVALUATION_TYPE $@ &
#python Stubborn/collect.py --sem_gpu_id 0 --exp_name rn_a400_sms10 --print_images 1 --sem_pred_prob_thr 0.9 --smp_step 10 --switch_step 19 --alpha 400 --col_rad 3.5 --start_ep 0 --escape 0 --num_sem_categories 23 --global_downscaling 3 --map_size_cm 4800 --evaluation $AGENT_EVALUATION_TYPE $@ &

python Stubborn/collect_maps.py --sem_gpu_id 0 --start_ep 0 --end_ep 500 --evaluation $AGENT_EVALUATION_TYPE $@ &
python Stubborn/collect_maps.py --sem_gpu_id 0 --start_ep 500 --end_ep 1000 --evaluation $AGENT_EVALUATION_TYPE $@ &
python Stubborn/collect_maps.py --sem_gpu_id 0 --start_ep 1000 --end_ep 1500 --evaluation $AGENT_EVALUATION_TYPE $@ &
python Stubborn/collect_maps.py --sem_gpu_id 0 --start_ep 1500 --end_ep 2000 --evaluation $AGENT_EVALUATION_TYPE $@ &
python Stubborn/collect_maps.py --sem_gpu_id 0 --start_ep 2000 --end_ep 2500 --evaluation $AGENT_EVALUATION_TYPE $@ &
python Stubborn/collect_maps.py --sem_gpu_id 0 --start_ep 2500 --end_ep 3000 --evaluation $AGENT_EVALUATION_TYPE $@ &
python Stubborn/collect_maps.py --sem_gpu_id 0 --start_ep 3000 --end_ep 3500 --evaluation $AGENT_EVALUATION_TYPE $@ &
python Stubborn/collect_maps.py --sem_gpu_id 0 --start_ep 3500 --end_ep 4000 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_maps.py --sem_gpu_id 4 --start_ep 8000 --end_ep 9000 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_maps.py --sem_gpu_id 4 --start_ep 9000 --end_ep 10000 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_maps.py --sem_gpu_id 5 --start_ep 10000 --end_ep 11000 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_maps.py --sem_gpu_id 5 --start_ep 11000 --end_ep 12000 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_maps.py --sem_gpu_id 6 --start_ep 12000 --end_ep 13000 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_maps.py --sem_gpu_id 6 --start_ep 13000 --end_ep 14000 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_maps.py --sem_gpu_id 7 --start_ep 14000 --end_ep 15000 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_maps.py --sem_gpu_id 7 --start_ep 15000 --end_ep 16000 --evaluation $AGENT_EVALUATION_TYPE $@ &

wait
