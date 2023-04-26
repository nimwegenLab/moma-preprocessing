#!/usr/bin/env bash

./build_docker_image.sh

HOST_SOURCE_PATH=""
CONTAINER_TARGET_PATH=""

docker run --rm -it --mount  type=bind,source=$HOST_SOURCE_PATH,target=$CONTAINER_TARGET_PATH mmpreprocesspy:v0.2.0
