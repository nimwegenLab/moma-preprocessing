#!/usr/bin/env bash

export MMPRE_HOME="/home/micha/Documents/01_work/git/mmpreprocesspy" # this will have to be removed at some point; should be set by the 'module load command'
export MM_PYTHON_ENVIRONMENT_PATH=mmpreprocesspy

# Create output folder
DATASET_DIR="/home/micha/Documents/01_work/git/MM_Testing/30__lis_20210813"
OUTPUT_DIR="$DATASET_DIR/output"
mkdir -p $OUTPUT_DIR
LOGFILE="$OUTPUT_DIR/output.log"

printf "DATASET_DIR: $DATASET_DIR"
printf "OUTPUT_DIR: $OUTPUT_DIR"

RAW_PATH="/home/micha/Documents/01_work/git/MM_Testing/30__lis_20210813/MMStack/20210813_VNG1040_AB2h_2h_1_MMStack.ome.tif"
PREPROC_DIR_TPL="$OUTPUT_DIR/mmpreproc_%s/"
GL_DETECTION_TEMPLATE_PATH="$DATASET_DIR/TEMPLATE/template_config.json"
# FLATFIELD_PATH="/scicore/home/nimwegen/alabal0000/MM_Data/Lis/20210813/20210813_gfpflatfield/20210813_gfpflatfield_2_MMStack.ome.tif"
POS_NAMES=(Pos0) # These are positions names
ROTATION=90.3
TMAX=5 # uncomment to specify the last frame to analyze
ROI_BOUNDARY_OFFSET_AT_MOTHER_CELL=0
NORMALIZATION_CONFIG_PATH="true"
NORMALIZATION_REGION_OFFSET=30

# generate an array of same length as position array, with every element containing the ROTATION scalar value
ROTATIONS=$ROTATION
for f in `seq ${#POS_NAMES[*]}`
do
        f=`echo $f - 1 | bc`
        ROTATIONS[$f]=$ROTATION
done

source $MMPRE_HOME/mm_dispatch_preprocessing.sh
mm_dispatch_preprocessing
