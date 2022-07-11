#!/usr/bin/env bash

python Stubborn/collect.py --timestep_limit 500 --evaluation $AGENT_EVALUATION_TYPE $@

