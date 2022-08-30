#!/usr/bin/env bash

# Switch step: explore -> smp
python Stubborn/eval.py -v 0 --sem_pred_prob_thr 0.9 --smp_step 10 --switch_step 19 --alpha 400 --col_rad 4 --global_downscaling 3 --map_size_cm 4800  --evaluation remote $@