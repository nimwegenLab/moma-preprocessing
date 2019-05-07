#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

#rm -rf $DIR/venv-testenv/

PREPROC_PACKAGE_PATH=$DIR/mmpreprocesspy

if [[ ! -z "$(command -v module)" ]]; then # are we running on SciCore?
  module purge
#  module load Python/3.5.2-goolf-1.7.20
#  module load Python/3.6.6-foss-2018b
else
  echo "Running without LMOD ..."
  # needs ubuntu packages: python3.7, python3.7-venv
fi

python3 -m venv $DIR/venv-testenv
#virtualenv --python=python3.6 $DIR/venv-testenv
source "$DIR/venv-testenv/bin/activate"

pip install wheel
pip install ipdb
pip install pudb
pip install exifread
pip install pillow
pip install scipy
pip install --editable $PREPROC_PACKAGE_PATH

#if [[ "$HOSTNAME" -eq "ierbert" ]]; then
echo "Copying OpenCV from the resource folder ..."
cp -a resources/cv2 venv-testenv/lib/python3.7/site-packages/
#else
#  echo "Installing OpenCV using pip ..."
##  pip install opencv-contrib-python
#fi

deactivate
