#!/usr/bin/env bash

###
# This script builds the docker image to containerize the mmpreprocesspy package.
###

source helpers.sh

PREV_IMAGE_TAG=$(get_image_tag)

# Get previously existing image with same tag to remove it at end of script and avoid dangling images.
PREV_IMAGE=$(docker images --filter=reference="${PREV_IMAGE_TAG}" -q)

docker build .. -t "${PREV_IMAGE_TAG}"

# Remove previous image with the same tag.
if [[ -z "${PREV_IMAGE}" ]]; then
  docker rmi "${PREV_IMAGE}"
fi