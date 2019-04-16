DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

#rm -rf $DIR/venv-testenv/

PACKAGE_PATH=$DIR/mmpreprocesspy

ml purge
ml Python/3.5.2-goolf-1.7.20

if [[ ! -d "$DIR/venv-testenv"  ]]; then
  printf "Venv not found. Setting up from scratch ...\n"
  virtualenv $DIR/venv-testenv
  source $DIR/venv-testenv/bin/activate

  pip install ipdb
  pip install pudb
  pip install exifread

  cd $DIR/mmpreprocesspy
  pip install --editable $PACKAGE_PATH
else
  printf "Venv exists. Only updating mmpreprocesspy ...\n"
  source $DIR/venv-testenv/bin/activate
  pip install --editable $PACKAGE_PATH --upgrade
fi

