#!/usr/bin/env bash

# Switch step: explore -> smp
python Stubborn/eval.py -v 0 --inhib_mode 2 --num_sem_categories 16 --dd_erode 1 --evaluation remote $@