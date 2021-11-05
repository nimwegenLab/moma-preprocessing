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

conda create -y --prefix $DIR/mmpreprocesspy_conda_environment python=3.7
source activate $DIR/mmpreprocesspy_conda_environment

######### packages for deep leaning ##############
conda install pip

conda install -y wheel
conda install -y -c conda-forge ipdb
conda install -y -c conda-forge pudb
conda install -y -c conda-forge exifread
conda install -y pillow
conda install -y scipy
conda install -y -c lightsource2-tag pystackreg
conda install -y conda-build
conda install -y scikit-image
conda install -y tifffile=2020.10.1
conda install -y zarr
conda install -y -c conda-forge imagecodecs
conda install -y pandas
#if [[ "$runningOnScicore" = true ]]; then
#    conda install $PREPROC_PACKAGE_PATH
#else
#    conda develop $PREPROC_PACKAGE_PATH
#fi
conda develop $PREPROC_PACKAGE_PATH

#conda install -y -c conda-forge opencv-contrib-python


pip install opencv-contrib-python

#if [[ "$runningOnScicore" = false  ]]; then
#  echo "Copying OpenCV from the resource folder ..."
#  cp -a resources/cv2 venv-testenv/lib/python3.7/site-packages/
#else
#  echo "Installing OpenCV using pip ..."
#  pip install opencv-contrib-python
#fi


conda deactivate
