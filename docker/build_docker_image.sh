#!/usr/bin/env bash

###
# This script builds the docker image to containerize the mmpreprocesspy package.
###


CONTAINER_TAG="mmpreprocesspy:v0.2.0"

# Get previously existing image with same tag to remove it at end of script and avoid dangling images.
PREV_IMAGE=$(docker images --filter=reference="${CONTAINER_TAG}" -q)

docker build .. -t ${CONTAINER_TAG}

# Remove previous image with the same tag.
if [[ -z "${PREV_IMAGE}" ]]; then
  docker rmi "${PREV_IMAGE}"
fi