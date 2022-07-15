#!/usr/bin/env bash

python Stubborn/collect.py --timestep_limit 1000 --evaluation $AGENT_EVALUATION_TYPE $@

