#!/usr/bin/env bash

eval $(conda shell.bash hook)

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
conda create -y --file $DIR/conda_package_list.txt -n mmpreprocesspy_conda_environment python=3.7
source activate mmpreprocesspy_conda_environment

conda install -y pip
pip install opencv-contrib-python==4.5.5.64
conda deactivate
