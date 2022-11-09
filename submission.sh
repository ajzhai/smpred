#!/usr/bin/env bash

# Switch step: explore -> smp
python Stubborn/eval.py -v 0 --inhib_mode 2  --alpha 100 --evaluation remote $@
#python Stubborn/eval.py -v 0 --inhib_mode 2  --alpha 100 --erode_recover 1 --map_size_cm 7200 --global_downscaling 3 --num_sem_categories 23 --evaluation remote $@