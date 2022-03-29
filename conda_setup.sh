#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

PREPROC_PACKAGE_PATH=$DIR/mmpreprocesspy

if [[ ! -z "$(command -v module)" ]]; then # are we running on SciCore?
  runningOnScicore=true
else
  runningOnScicore=false
fi

#if [[ "$runningOnScicore" = true  ]]; then
#  module purge
#  #  module load Python/3.5.2-goolf-1.7.20
#  #  module load Python/3.6.6-foss-2018b
#  module load Python/3.7.3-foss-2018b
#else
#  echo "Running without LMOD ..."
#    # needs ubuntu packages: python3.7, python3.7-venv
#fi

#python3 -m venv $DIR/venv
#source "$DIR/venv/bin/activate"

eval $(conda shell.bash hook)

conda create --file $DIR/conda_package_list.txt -y --prefix $DIR/mmpreprocesspy_conda_environment python=3.7
source activate $DIR/mmpreprocesspy_conda_environment

conda install pip
pip install opencv-contrib-python==4.5.5.64

#if [[ "$runningOnScicore" = true ]]; then
#    conda install $PREPROC_PACKAGE_PATH
#else
#    conda develop $PREPROC_PACKAGE_PATH
#fi
conda develop $PREPROC_PACKAGE_PATH

conda deactivate
