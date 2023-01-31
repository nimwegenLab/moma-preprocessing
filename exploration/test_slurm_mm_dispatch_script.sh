#!/usr/bin/env bash

export MMPRE_HOME="/home/micha/Documents/01_work/git/mmpreprocesspy" # this will have to be removed at some point; should be set by the 'module load command'
export MM_PYTHON_ENVIRONMENT_PATH=mmpreprocesspy

# Create output folder
DATASET_DIR="/media/micha/T7/data_michael_mell/preprocessing_test_data/MM_Testing/35__lis__20220320__repeat"
OUTPUT_DIR="$DATASET_DIR/test_bash_run_output"
mkdir -p $OUTPUT_DIR
LOGFILE="$OUTPUT_DIR/output.log"

printf "DATASET_DIR: $DATASET_DIR\n"
printf "OUTPUT_DIR: $OUTPUT_DIR\n"

RAW_PATH="$DATASET_DIR/MMSTACK/20220320_VNG1040_AB2h_1_Frame0-478_resaved/20220320_VNG1040_AB2h_1_Frame0-478_resaved_MMStack_Pos0.ome.tif"
PREPROC_DIR_TPL="$OUTPUT_DIR/mmpreproc_%s/"
GL_DETECTION_TEMPLATE_PATH="$DATASET_DIR/TEMPLATE_MICHAEL/template_config.json"
# FLATFIELD_PATH="/scicore/home/nimwegen/alabal0000/MM_Data/Lis/20210813/20210813_gfpflatfield/20210813_gfpflatfield_2_MMStack.ome.tif"
POS_NAMES=(Pos0) # These are positions names
FRAMES_TO_IGNORE=(2,4)
ROTATION=90.3
TMAX=5 # uncomment to specify the last frame to analyze
ROI_BOUNDARY_OFFSET_AT_MOTHER_CELL=0
NORMALIZATION_CONFIG_PATH="true"
NORMALIZATION_REGION_OFFSET=30
IMAGE_REGISTRATION_METHOD=2

# generate an array of same length as position array, with every element containing the ROTATION scalar value
ROTATIONS=$ROTATION
for f in `seq ${#POS_NAMES[*]}`
do
        f=`echo $f - 1 | bc`
        ROTATIONS[$f]=$ROTATION
done

source $MMPRE_HOME/mm_dispatch_preprocessing.sh
mm_dispatch_preprocessing
