#!/usr/bin/env bash

eval $(conda shell.bash hook)

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PREPROC_PACKAGE_PATH=$DIR/mmpreprocesspy

source activate mmpreprocesspy_conda_environment
conda develop $PREPROC_PACKAGE_PATH
conda deactivate