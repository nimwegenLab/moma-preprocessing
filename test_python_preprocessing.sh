#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"


export MMPRE_HOME=$DIR # this will have to be removed at some point; should be set by the 'module load command'
export MM_PYTHON_ENVIRONMENT_PATH=$DIR/venv-testenv

# Create output folder
DATASET_DIR="/home/micha/Documents/01_work/git/MM_Testing/15_lis_20201119_VNG1040_AB2h_2h_1/MMStack"
OUTPUT_DIR="$DATASET_DIR/output"
mkdir -p $OUTPUT_DIR
LOGFILE="$OUTPUT_DIR/output.log"
echo ========== $(date) ========== >> output.log

printf "DATASET_DIR: $DATASET_DIR"
printf "OUTPUT_DIR: $OUTPUT_DIR"

#RAW_PATH="/scicore/home/nimwegen/GROUP/MM_Data/Guillaume/20180723_GW277_glucose37/20180723_GW277_glucose37_1/"
RAW_PATH="/home/micha/Documents/01_work/git/MM_Testing/15_lis_20201119_VNG1040_AB2h_2h_1/MMStack/"
#PREPROC_DIR_TPL="$OUTPUT_DIR/20180723_GW277_glucose37_1_slurm_%s/"
PREPROC_DIR_TPL="$OUTPUT_DIR/15_lis_20201119_VNG1040_AB2h_2h_1/"
POS_NAMES=(Pos0) # These are positions names
ROTATIONS=(180)
TMAX=10 # uncomment to specify the last frame to analyze
ROI_BOUNDARY_OFFSET_AT_MOTHER_CELL=20


source $DIR/mm_dispatch_preprocessing.sh
mm_dispatch_preprocessing

