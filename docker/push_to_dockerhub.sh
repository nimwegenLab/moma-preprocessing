#!/usr/bin/env bash

source helpers.sh

docker push $(get_image_tag)
