#!/usr/bin/env bash

# Switch step: explore -> smp
python Stubborn/eval.py -v 0 --sem_pred_prob_thr 0.9 --smp_step 10 --switch_step 19 --alpha 400 --col_rad 4 --sf_thr '-1' --goal_erode 2 --segformer 1 --evaluation remote $@