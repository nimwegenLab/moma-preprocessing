#!/usr/bin/env bash

export MMPRE_HOME="/home/micha/Documents/01_work/git/mmpreprocesspy" # this will have to be removed at some point; should be set by the 'module load command'
export MM_PYTHON_ENVIRONMENT_PATH=mmpreprocesspy

# Create output folder
DATASET_DIR="/media/micha/T7/data_michael_mell/preprocessing_test_data/MM_Testing/38__bor_20230401"
OUTPUT_DIR="$DATASET_DIR/test_bash_run_output"
mkdir -p $OUTPUT_DIR
LOGFILE="$OUTPUT_DIR/output.log"

printf "DATASET_DIR: $DATASET_DIR\n"
printf "OUTPUT_DIR: $OUTPUT_DIR\n"

RAW_PATH="$DATASET_DIR/MMSTACK/20230401_starvation_naturalisolates_med3/20230401_nat_iso_carbon_med3_1_MMStack.ome.tif"
 FLATFIELD_PATH="$DATASET_DIR/MMSTACK/20230401_flatfield_1/20230401_flatfield_1_MMStack.ome.tif"
PREPROC_DIR_TPL="$OUTPUT_DIR/mmpreproc_%s/"
GL_DETECTION_TEMPLATE_PATH="$DATASET_DIR/GL_DETECTION_TEMPLATE/template_20230401.json"

#POS_NAMES=(1-Pos000 1-Pos001 1-Pos002 1-Pos003 2-Pos000 2-Pos001 2-Pos002 2-Pos003 3-Pos000 3-Pos001 3-Pos002 3-Pos003 5-Pos000 5-Pos001 5-Pos002 5-Pos003 6-Pos000 6-Pos001 6-Pos002 6-Pos003 7-Pos000 7-Pos001 7-Pos002 7-Pos003 8-Pos000 8-Pos001 8-Pos002 8-Pos003 9-Pos000 9-Pos001 9-Pos002 9-Pos003)

POS_NAMES=(1-Pos000 1-Pos002) # These are positions names

ROTATION=90.1

TMAX=5 # uncomment to specify the last frame to analyze

ROI_BOUNDARY_OFFSET_AT_MOTHER_CELL=0
NORMALIZATION_CONFIG_PATH="true"
NORMALIZATION_REGION_OFFSET=30
#FRAMES_TO_IGNORES=()

# generate an array of same length as position array, with every element containing the ROTATION scalar value
ROTATIONS=$ROTATION
for f in `seq ${#POS_NAMES[*]}`
do
        f=`echo $f - 1 | bc`
        ROTATIONS[$f]=$ROTATION
done

source $MMPRE_HOME/mm_dispatch_preprocessing.sh
mm_dispatch_preprocessing
