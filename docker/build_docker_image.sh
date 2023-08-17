#!/usr/bin/env bash

###
# This script builds the docker image to containerize the mmpreprocesspy package.
###

source helpers.sh

IMAGE_TAG=$(get_image_tag)

# Get previously existing image with same tag to remove it at end of script and avoid dangling images.
PREV_IMAGES=$(docker images --filter=reference="${IMAGE_TAG}" -q)

docker build .. -t "${IMAGE_TAG}"

# Remove previous image with the same tag.
if [[ -z "${PREV_IMAGES}" ]]; then
  docker rmi "${PREV_IMAGES}"
fi