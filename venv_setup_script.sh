#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

#rm -rf $DIR/venv-testenv/

PREPROC_PACKAGE_PATH=$DIR/mmpreprocesspy

if [[ ! -z "$(command -v module)" ]]; then # are we running on SciCore?
  module load purge
  module load Python/3.5.2-goolf-1.7.20
else
  echo "Running without LMOD ..."
  # needs ubuntu packages: python3.5, python3.5-venv
fi

virtualenv --python=python3.5 $DIR/venv-testenv
source "$DIR/venv-testenv/bin/activate"

pip install ipdb
pip install pudb
pip install exifread
pip install pillow
pip install scipy
pip install --editable $PREPROC_PACKAGE_PATH

deactivate
