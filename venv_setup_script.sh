DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

#rm -rf $DIR/venv-testenv/

ml purge
ml Python/3.5.2-goolf-1.7.20

if [[ ! -d "$DIR/venv-testenv"  ]]; then
  printf "Venv not found. Setting up from scratch."
  virtualenv $DIR/venv-testenv
  source $DIR/venv-testenv/bin/activate

  pip install ipdb
  pip install pudb
  pip install exifread

  cd $DIR/mmpreprocesspy
  pip install .
else
  printf "Venv exists. Only updating mmpreprocesspy ..."
  source $DIR/venv-testenv/bin/activate
  cd mmpreprocesspy
  pip install . --upgrade
  cd -
fi
