#!/usr/bin/env bash

source activate mmpreprocesspy_conda_environment
eval "python /preprocessing/call_preproc_fun.py" "$@"
conda deactivate