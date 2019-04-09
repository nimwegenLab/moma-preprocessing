DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

rm -rf $DIR/venv-testenv/

ml purge
ml Python/3.5.2-goolf-1.7.20
virtualenv venv-testenv
source venv-testenv/bin/activate
cd colicycle
pip install .

