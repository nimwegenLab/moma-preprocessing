#!/usr/bin/env bash

echo "Hello world!"
echo "These were the arguments: " "$@"

#conda activate mmpreprocesspy_conda_environment
source activate mmpreprocesspy_conda_environment

#python /preprocessing/call_preproc_fun.py -i "/data/input_path" -o "/data/output_path" "$@"

echo

echo "/preprocessing/call_preproc_fun.sh" "$@"

echo

python /preprocessing/call_preproc_fun.py "$@"

conda deactivate