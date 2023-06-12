#!/usr/bin/env bash

source helpers.sh

echo docker tag $(get_image_tag) $(get_docker_hub_tag)
echo docker push $(get_docker_hub_tag)
