#!/usr/bin/env bash

# Switch step: explore -> smp
python Stubborn/eval.py -v 0 --inhib_mode 0 --num_sem_categories 23 --evaluation remote $@