#!/usr/bin/env bash

./build_docker_image.sh

./containerized_test_slurm_mm_dispatch_script.sh

#HOST_SOURCE_PATH="/media/micha/T7/data_michael_mell/preprocessing_test_data/MM_Testing/35__lis__20220320__repeat"
#CONTAINER_TARGET_PATH="/data"

#docker run --rm -it --mount  type=bind,source=$HOST_SOURCE_PATH,target=$CONTAINER_TARGET_PATH mmpreprocesspy:v0.2.0
