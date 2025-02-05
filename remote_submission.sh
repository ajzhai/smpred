#!/usr/bin/env bash

# python Stubborn/collect_slam.py --sem_gpu_id 0 --exp_name debug_dn --print_images 1 --alpha 100 --overwrite_map 1 --depth_noise_mult 0.1 --use_edw 1 --end_ep 1 --evaluation $AGENT_EVALUATION_TYPE $@ &

# python Stubborn/collect_slam.py -v 0 --dump_location ./data/tmp --exp_name slam2 --print_images 1 --alpha 100 --pose_noise_std 0.0 --switch_step 0 --start_ep 4 --end_ep 5 --evaluation $AGENT_EVALUATION_TYPE $@ 
#python Stubborn/collect_mp3d.py --dump_location ./data/tmp --exp_name vis_mp3d --print_images 1 --start_ep 7 --end_ep 8 --evaluation $AGENT_EVALUATION_TYPE $@  # Stubborn (rednet)

# python Stubborn/collect_withgt.py --sem_gpu_id 0 --exp_name debug_gt --print_images 1 --alpha 100 --start_ep 14 --end_ep 15 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect.py --sem_gpu_id 0 --exp_name vis_seed100 --print_images 1 --alpha 100 --end_ep 100 --evaluation $AGENT_EVALUATION_TYPE $@ &
python Stubborn/collect_seg.py --sem_gpu_id 0 --exp_name segnewval --print_images 0 --evaluation $AGENT_EVALUATION_TYPE $@ &

# python Stubborn/collect_slam.py -v 0 --sem_gpu_id 1 --exp_name dn1_1 --overwrite_map 1 --depth_noise_mult 0.1 --print_images 0 --alpha 100 --switch_step 0 --start_ep 0 --end_ep 100 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_slam.py -v 0 --sem_gpu_id 1 --exp_name dn1_2 --overwrite_map 1 --depth_noise_mult 0.1 --print_images 0 --alpha 100 --switch_step 0 --start_ep 100 --end_ep 200 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_slam.py -v 0 --sem_gpu_id 1 --exp_name dn1_3 --overwrite_map 1 --depth_noise_mult 0.1 --print_images 0 --alpha 100 --switch_step 0 --start_ep 200 --end_ep 300 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_slam.py -v 0 --sem_gpu_id 1 --exp_name dn1_4 --overwrite_map 1 --depth_noise_mult 0.1 --print_images 0 --alpha 100 --switch_step 0 --start_ep 300 --end_ep 400 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_slam.py -v 0 --sem_gpu_id 1 --exp_name dn1_5 --overwrite_map 1 --depth_noise_mult 0.1 --print_images 0 --alpha 100 --switch_step 0 --start_ep 400 --end_ep 500 --evaluation $AGENT_EVALUATION_TYPE $@ &

# python Stubborn/collect_slam.py -v 0 --sem_gpu_id 3 --exp_name dn2_1 --overwrite_map 1 --depth_noise_mult 0.2 --print_images 0 --alpha 100 --switch_step 0 --start_ep 0 --end_ep 100 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_slam.py -v 0 --sem_gpu_id 3 --exp_name dn2_2 --overwrite_map 1 --depth_noise_mult 0.2 --print_images 0 --alpha 100 --switch_step 0 --start_ep 100 --end_ep 200 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_slam.py -v 0 --sem_gpu_id 3 --exp_name dn2_3 --overwrite_map 1 --depth_noise_mult 0.2 --print_images 0 --alpha 100 --switch_step 0 --start_ep 200 --end_ep 300 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_slam.py -v 0 --sem_gpu_id 3 --exp_name dn2_4 --overwrite_map 1 --depth_noise_mult 0.2 --print_images 0 --alpha 100 --switch_step 0 --start_ep 300 --end_ep 400 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_slam.py -v 0 --sem_gpu_id 3 --exp_name dn2_5 --overwrite_map 1 --depth_noise_mult 0.2 --print_images 0 --alpha 100 --switch_step 0 --start_ep 400 --end_ep 500 --evaluation $AGENT_EVALUATION_TYPE $@ &

# python Stubborn/collect_slam.py -v 0 --sem_gpu_id 4 --exp_name dn3_1 --overwrite_map 1 --depth_noise_mult 0.3 --print_images 0 --alpha 100 --switch_step 0 --start_ep 0 --end_ep 100 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_slam.py -v 0 --sem_gpu_id 4 --exp_name dn3_2 --overwrite_map 1 --depth_noise_mult 0.3 --print_images 0 --alpha 100 --switch_step 0 --start_ep 100 --end_ep 200 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_slam.py -v 0 --sem_gpu_id 4 --exp_name dn3_3 --overwrite_map 1 --depth_noise_mult 0.3 --print_images 0 --alpha 100 --switch_step 0 --start_ep 200 --end_ep 300 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_slam.py -v 0 --sem_gpu_id 4 --exp_name dn3_4 --overwrite_map 1 --depth_noise_mult 0.3 --print_images 0 --alpha 100 --switch_step 0 --start_ep 300 --end_ep 400 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_slam.py -v 0 --sem_gpu_id 4 --exp_name dn3_5 --overwrite_map 1 --depth_noise_mult 0.3 --print_images 0 --alpha 100 --switch_step 0 --start_ep 400 --end_ep 500 --evaluation $AGENT_EVALUATION_TYPE $@ &

# python Stubborn/collect_slam.py -v 0 --sem_gpu_id 0 --exp_name dn4_1 --overwrite_map 1 --depth_noise_mult 0.4 --print_images 0 --alpha 100 --switch_step 0 --start_ep 0 --end_ep 100 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_slam.py -v 0 --sem_gpu_id 0 --exp_name dn4_2 --overwrite_map 1 --depth_noise_mult 0.4 --print_images 0 --alpha 100 --switch_step 0 --start_ep 100 --end_ep 200 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_slam.py -v 0 --sem_gpu_id 0 --exp_name dn4_3 --overwrite_map 1 --depth_noise_mult 0.4 --print_images 0 --alpha 100 --switch_step 0 --start_ep 200 --end_ep 300 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_slam.py -v 0 --sem_gpu_id 0 --exp_name dn4_4 --overwrite_map 1 --depth_noise_mult 0.4 --print_images 0 --alpha 100 --switch_step 0 --start_ep 300 --end_ep 400 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_slam.py -v 0 --sem_gpu_id 0 --exp_name dn4_5 --overwrite_map 1 --depth_noise_mult 0.4 --print_images 0 --alpha 100 --switch_step 0 --start_ep 400 --end_ep 500 --evaluation $AGENT_EVALUATION_TYPE $@ &

# python Stubborn/collect_slam.py -v 0 --sem_gpu_id 2 --exp_name dn5_1 --overwrite_map 1 --depth_noise_mult 0.5 --print_images 0 --alpha 100 --switch_step 0 --start_ep 0 --end_ep 100 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_slam.py -v 0 --sem_gpu_id 2 --exp_name dn5_2 --overwrite_map 1 --depth_noise_mult 0.5 --print_images 0 --alpha 100 --switch_step 0 --start_ep 100 --end_ep 200 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_slam.py -v 0 --sem_gpu_id 2 --exp_name dn5_3 --overwrite_map 1 --depth_noise_mult 0.5 --print_images 0 --alpha 100 --switch_step 0 --start_ep 200 --end_ep 300 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_slam.py -v 0 --sem_gpu_id 2 --exp_name dn5_4 --overwrite_map 1 --depth_noise_mult 0.5 --print_images 0 --alpha 100 --switch_step 0 --start_ep 300 --end_ep 400 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_slam.py -v 0 --sem_gpu_id 2 --exp_name dn5_5 --overwrite_map 1 --depth_noise_mult 0.5 --print_images 0 --alpha 100 --switch_step 0 --start_ep 400 --end_ep 500 --evaluation $AGENT_EVALUATION_TYPE $@ &




# python Stubborn/collect_slam.py -v 0 --sem_gpu_id 1 --exp_name gridslam0h --print_images 0 --alpha 100 --switch_step 0 --start_ep 0 --end_ep 100 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_slam.py -v 0 --sem_gpu_id 0 --exp_name gridslam1n --print_images 0 --alpha 100 --switch_step 0 --start_ep 100 --end_ep 200 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_slam.py -v 0 --sem_gpu_id 1 --exp_name gridslam2h --print_images 0 --alpha 100 --switch_step 0 --start_ep 200 --end_ep 300 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_slam.py -v 0 --sem_gpu_id 1 --exp_name gridslam3h --print_images 0 --alpha 100 --switch_step 0 --start_ep 300 --end_ep 400 --evaluation $AGENT_EVALUATION_TYPE $@ &

# python Stubborn/collect_slam.py -v 0 --sem_gpu_id 5 --exp_name icpslam0 --icp_refine 1 --alpha 100 --switch_step 0 --start_ep 0 --end_ep 100 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_slam.py -v 0 --sem_gpu_id 5 --exp_name icpslam1 --icp_refine 1 --alpha 100 --switch_step 0 --start_ep 100 --end_ep 200 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_slam.py -v 0 --sem_gpu_id 5 --exp_name icpslam2 --icp_refine 1 --alpha 100 --switch_step 0 --start_ep 200 --end_ep 300 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_slam.py -v 0 --sem_gpu_id 5 --exp_name icpslam3 --icp_refine 1 --alpha 100 --switch_step 0 --start_ep 300 --end_ep 400 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_slam.py -v 0 --sem_gpu_id 5 --exp_name icpslam4 --icp_refine 1 --alpha 100 --switch_step 0 --start_ep 400 --end_ep 500 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect.py -v 0 --sem_gpu_id 7  --exp_name pn05 --alpha 100 --pose_noise_std 0.05 --switch_step 0 --evaluation $AGENT_EVALUATION_TYPE $@  &

# python Stubborn/collect.py -v 0 --sem_gpu_id 7  --exp_name pn05owm --alpha 100 --pose_noise_std 0.05 --overwrite_map 1 --switch_step 0 --evaluation $AGENT_EVALUATION_TYPE $@  &
# python Stubborn/collect.py -v 0 --sem_gpu_id 7  --exp_name pn10owm --alpha 100 --pose_noise_std 0.10 --overwrite_map 1 --switch_step 0 --evaluation $AGENT_EVALUATION_TYPE $@  &
# python Stubborn/collect.py -v 0 --sem_gpu_id 7  --exp_name pn15owm --alpha 100 --pose_noise_std 0.15 --overwrite_map 1 --switch_step 0 --evaluation $AGENT_EVALUATION_TYPE $@  &
# python Stubborn/collect.py -v 0 --sem_gpu_id 7  --exp_name pn20owm --alpha 100 --pose_noise_std 0.20 --overwrite_map 1 --switch_step 0 --evaluation $AGENT_EVALUATION_TYPE $@  &
# python Stubborn/collect.py -v 0 --sem_gpu_id 0  --exp_name pn25owm --alpha 100 --pose_noise_std 0.25 --overwrite_map 1 --switch_step 0 --evaluation $AGENT_EVALUATION_TYPE $@  &
# python Stubborn/collect.py -v 0 --sem_gpu_id 1  --exp_name pn30owm --alpha 100 --pose_noise_std 0.30 --overwrite_map 1 --switch_step 0 --evaluation $AGENT_EVALUATION_TYPE $@  &

#python Stubborn/collect.py -v 0 --sem_gpu_id 0  --exp_name defa_rat1 --print_images 0 --switch_step 19  --evaluation $AGENT_EVALUATION_TYPE $@  &
#python Stubborn/collect_maps_gt.py -v 0 --print_images 0 --evaluation $AGENT_EVALUATION_TYPE $@ 

# python Stubborn/collect_stubborn.py -v 0 --sem_gpu_id 1 --exp_name mp3d_val0 --print_images 0 --start_ep 0 --end_ep 695  --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_stubborn.py -v 0 --sem_gpu_id 1 --exp_name mp3d_val1 --print_images 0 --start_ep 695 --end_ep 1195  --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_stubborn.py -v 0 --sem_gpu_id 1 --exp_name mp3d_val2 --print_images 0 --start_ep 1195 --end_ep 1695 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_stubborn.py -v 0 --sem_gpu_id 1 --exp_name mp3d_val3 --print_images 0 --start_ep 1695 --end_ep 2195 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_withgt.py -v 0 --sem_gpu_id 1 --exp_name gt --print_images 0 --alpha 250 --inhib_mode 0 --evaluation $AGENT_EVALUATION_TYPE $@ &

# python Stubborn/collect_mp3d.py -v 0 --sem_gpu_id 0 --exp_name mp3d_val0_a100_s0 --print_images 0 --start_ep 0 --end_ep 695 --alpha 100 --erode_recover 1 --inhib_mode 0 --switch_step 0 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_mp3d.py -v 0 --sem_gpu_id 0 --exp_name mp3d_val1_a100_s0 --print_images 0 --start_ep 695 --end_ep 1195 --alpha 100 --erode_recover 1 --inhib_mode 0 --switch_step 0 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_mp3d.py -v 0 --sem_gpu_id 0 --exp_name mp3d_val2_a100_s0 --print_images 0 --start_ep 1195 --end_ep 1695 --alpha 100 --erode_recover 1 --inhib_mode 0 --switch_step 0 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_mp3d.py -v 0 --sem_gpu_id 0 --exp_name mp3d_val3_a100_s0 --print_images 0 --start_ep 1695 --end_ep 2195 --alpha 100 --erode_recover 1 --inhib_mode 0 --switch_step 0 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_withgt.py -v 0 --sem_gpu_id 0 --exp_name gt_s0 --print_images 0 --alpha 250 --switch_step 0 --inhib_mode 0 --evaluation $AGENT_EVALUATION_TYPE $@ &

# python Stubborn/collect_mp3d.py -v 0 --sem_gpu_id 2 --exp_name mp3d_val0_a150 --print_images 0 --start_ep 0 --end_ep 695 --alpha 150 --erode_recover 1 --inhib_mode 0  --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_mp3d.py -v 0 --sem_gpu_id 2 --exp_name mp3d_val1_a150 --print_images 0 --start_ep 695 --end_ep 1195 --alpha 150 --erode_recover 1 --inhib_mode 0 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_mp3d.py -v 0 --sem_gpu_id 2 --exp_name mp3d_val2_a150 --print_images 0 --start_ep 1195 --end_ep 1695 --alpha 150 --erode_recover 1 --inhib_mode 0  --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_mp3d.py -v 0 --sem_gpu_id 2 --exp_name mp3d_val3_a150 --print_images 0 --start_ep 1695 --end_ep 2195 --alpha 150 --erode_recover 1 --inhib_mode 0  --evaluation $AGENT_EVALUATION_TYPE $@ &

# python Stubborn/collect_mp3d.py -v 0 --sem_gpu_id 3 --exp_name mp3d_val0_a250 --print_images 0 --start_ep 0 --end_ep 695 --alpha 250 --erode_recover 1 --inhib_mode 0  --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_mp3d.py -v 0 --sem_gpu_id 3 --exp_name mp3d_val1_a250 --print_images 0 --start_ep 695 --end_ep 1195 --alpha 250 --erode_recover 1 --inhib_mode 0 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_mp3d.py -v 0 --sem_gpu_id 3 --exp_name mp3d_val2_a250 --print_images 0 --start_ep 1195 --end_ep 1695 --alpha 250 --erode_recover 1 --inhib_mode 0  --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_mp3d.py -v 0 --sem_gpu_id 3 --exp_name mp3d_val3_a250 --print_images 0 --start_ep 1695 --end_ep 2195 --alpha 250 --erode_recover 1 --inhib_mode 0  --evaluation $AGENT_EVALUATION_TYPE $@ &

# python Stubborn/collect_mp3d.py -v 0 --sem_gpu_id 5 --exp_name mp3d_val0_a400 --print_images 0 --start_ep 0 --end_ep 695 --alpha 400 --erode_recover 1 --inhib_mode 0  --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_mp3d.py -v 0 --sem_gpu_id 5 --exp_name mp3d_val1_a400 --print_images 0 --start_ep 695 --end_ep 1195 --alpha 400 --erode_recover 1 --inhib_mode 0 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_mp3d.py -v 0 --sem_gpu_id 5 --exp_name mp3d_val2_a400 --print_images 0 --start_ep 1195 --end_ep 1695 --alpha 400 --erode_recover 1 --inhib_mode 0  --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_mp3d.py -v 0 --sem_gpu_id 5 --exp_name mp3d_val3_a400 --print_images 0 --start_ep 1695 --end_ep 2195 --alpha 400 --erode_recover 1 --inhib_mode 0  --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_withgt.py -v 0 --sem_gpu_id 5 --exp_name gt_a100 --print_images 0 --alpha 100 --inhib_mode 0 --evaluation $AGENT_EVALUATION_TYPE $@ &

# # python Stubborn/collect_mp3d.py -v 0 --sem_gpu_id 2 --exp_name mp3d_a125 --print_images 0 --alpha 125 --evaluation $AGENT_EVALUATION_TYPE $@ &
# # # python Stubborn/collect_mp3d.py -v 0 --sem_gpu_id 0 --exp_name mp3d_a250 --print_images 0 --alpha 250 --evaluation $AGENT_EVALUATION_TYPE $@ &
# # # python Stubborn/collect_mp3d.py -v 0 --sem_gpu_id 0 --exp_name mp3d_a300 --print_images 0 --alpha 300 --evaluation $AGENT_EVALUATION_TYPE $@ &

# python Stubborn/collect_mp3d.py -v 0 --sem_gpu_id 6 --exp_name mp3d_val0_inh2 --print_images 0 --start_ep 0 --end_ep 695 --alpha 100 --erode_recover 1 --inhib_mode 2 --map_size_cm 7200 --global_downscaling 3 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_mp3d.py -v 0 --sem_gpu_id 6 --exp_name mp3d_val1_inh2 --print_images 0 --start_ep 695 --end_ep 1195 --alpha 100 --erode_recover 1 --inhib_mode 2 --map_size_cm 7200 --global_downscaling 3 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_mp3d.py -v 0 --sem_gpu_id 6 --exp_name mp3d_val2_inh2 --print_images 0 --start_ep 1195 --end_ep 1695 --alpha 100 --erode_recover 1 --inhib_mode 2 --map_size_cm 7200 --global_downscaling 3 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_mp3d.py -v 0 --sem_gpu_id 6 --exp_name mp3d_val3_inh2 --print_images 0 --start_ep 1695 --end_ep 2195 --alpha 100 --erode_recover 1 --inhib_mode 2 --map_size_cm 7200 --global_downscaling 3 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect_withgt.py -v 0 --sem_gpu_id 6 --exp_name gt_a800 --print_images 0 --alpha 800 --inhib_mode 0 --evaluation $AGENT_EVALUATION_TYPE $@ &

# python Stubborn/collect.py --sem_gpu_id 0 --exp_name final_a1 --print_images 0 --alpha 1 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect.py --sem_gpu_id 0 --exp_name final_a10 --print_images 0 --alpha 10 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect.py --sem_gpu_id 0 --exp_name final_a3200 --print_images 0 --alpha 3200 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect.py --sem_gpu_id 0 --exp_name final_nopred --print_images 0 --alpha 0 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect.py --sem_gpu_id 0 --exp_name final_noddwt --print_images 0 --alpha -1 --evaluation $AGENT_EVALUATION_TYPE $@ &
#python Stubborn/collect.py --sem_gpu_id 2 --exp_name final_nopreds --print_images 0 --alpha 0 --evaluation $AGENT_EVALUATION_TYPE $@ &


# python Stubborn/collect.py --sem_gpu_id 2 --exp_name test_dde1_inh0 --print_images 0 --dd_erode 1 --inhib_mode 0 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect.py --sem_gpu_id 2 --exp_name test_dde1_inh2 --print_images 0 --dd_erode 1 --inhib_mode 2 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect.py --sem_gpu_id 2 --exp_name new_nr_inh2_notpr --print_images 0 --inhib_mode 2 --evaluation $AGENT_EVALUATION_TYPE $@ &
# python Stubborn/collect.py --sem_gpu_id 7 --exp_name new_norecover --print_images 0 --erode_recover 0 --evaluation $AGENT_EVALUATION_TYPE $@ &

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
