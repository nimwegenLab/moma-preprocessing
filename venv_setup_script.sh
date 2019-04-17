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

if [[ ! -d "$DIR/venv-testenv"  ]]; then
  printf "Venv not found. Setting up from scratch ...\n"
  virtualenv --python=python3.5 $DIR/venv-testenv
  source "$DIR/venv-testenv/bin/activate"

  python3.5 -m pip install ipdb
  python3.5 -m pip install pudb
  python3.5 -m pip install exifread

  pip install --editable $PREPROC_PACKAGE_PATH
else
  printf "Venv exists. Only updating mmpreprocesspy ...\n"
  source "$DIR/venv-testenv/bin/activate"
  pip install --editable $PREPROC_PACKAGE_PATH --upgrade
fi

