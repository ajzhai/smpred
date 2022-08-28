#!/usr/bin/env bash

# Switch step: explore -> smp
python Stubborn/eval.py -v 0 --smp_step 10 --switch_step 19 --alpha 400 --timestep_limit 1000 --evaluation remote $@