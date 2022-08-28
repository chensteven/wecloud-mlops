#!/bin/bash

export JOB_NAME="mlops_project_novice1"
export IMAGE="myimage2_news"
# export TAG="latest"
export PYTHON_ENV="development"
export API_PORT=80
export WORKERS=2
export TIMEOUT=300
export LOG_FOLDER=/var/log/ml-project1

echo ${IMAGE}
#echo ${IMAGE}:${TAG}

# Add your authentication command for the docker image registry here

# force pull and update the image, use this in remote host only
# docker pull ${IMAGE}:${TAG}

# stop running container with same job name, if any
if [ "$(docker ps -a | grep $JOB_NAME)" ]; then
  docker stop ${JOB_NAME} && docker rm ${JOB_NAME}
fi

# start docker container WITHOUT gpu and log volume
docker run -d \
  --rm \
  -p ${API_PORT}:80 \
  -e "WORKERS=${WORKERS}" \
  -e "TIMEOUT=${TIMEOUT}" \
  -e "PYTHON_ENV=${PYTHON_ENV}" \
  --name="${JOB_NAME}" \
  ${IMAGE}
  # ${IMAGE}:${TAG}