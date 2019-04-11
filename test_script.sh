DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

#ml purge
#ml Python/3.5.2-goolf-1.7.20
#source venv-testenv/bin/activate

#pip install ipdb
#pip install pudb
#pip install exifread

# Install updated colicycle python code
cd colicycle
pip install . --upgrade
cd -

mkdir -p $DIR/output

LOGFILE=$DIR/output/output.log
echo ========== $(date) ========== >> output.log

OUTPUT_DIR=$DIR/output

RAW_PATH="/scicore/home/nimwegen/GROUP/MM_Data/Guillaume/20180723_GW277_glucose37/20180723_GW277_glucose37_1/"
PREPROC_DIR_TPL="$OUTPUT_DIR/20180723_GW277_glucose37_1_slurm_%s/"
POS_NAMES=(Pos0) # These are positions names
ROTATIONS=(180)
TMAX=10 # uncomment to specify the last frame to analyze

export MMPRE_HOME=$DIR # this will have to be removed at some point; should be set by the 'module load command'

source $DIR/mm_dispatch_preprocessing_python.sh
mm_dispatch_preprocessing_python

