#!/usr/bin/env bash

DOCKER_NAME="remote_submission"

while [[ $# -gt 0 ]]
do
key="${1}"

case $key in
      --docker-name)
      shift
      DOCKER_NAME="${1}"
	  shift
      ;;
    *) # unknown arg
      echo unkown arg ${1}
      exit
      ;;
esac
done

docker run -v $(pwd)/habitat-challenge-data:/habitat-challenge-data \
    -v $(pwd)/data:/data\
    --gpus='"device=0,1,2,3"' \
    -e "AGENT_EVALUATION_TYPE=local" \
    -e "TRACK_CONFIG_FILE=/challenge_objectnav2022.local.rgbd.yaml" \
    ${DOCKER_NAME}\

