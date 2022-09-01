#!/usr/bin/env bash

# Switch step: explore -> smp
python Stubborn/eval.py -v 0 --escape 1 --toiletgrow 1 --sem_pred_prob_thr 0.9 --smp_step 10 --switch_step 19 --alpha 250 --global_downscaling 2 --map_size_cm 4800 --segformer 0 --num_sem_categories 16 --evaluation remote $@