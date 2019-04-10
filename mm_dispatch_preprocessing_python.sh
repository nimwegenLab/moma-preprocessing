# 
# Script dispatches the preprocessing of the dataset.
#


mm_dispatch_preprocessing(){
# load required module
#module load MMPreproc

MMPRE_EXIST=$(module av MMPreproc 2>&1 | grep MMPreproc | wc -l)
if [ $MMPRE_EXIST -eq 0 ]; then printf "The module MMPreproc cannot be found. Aborting...\n" >&2; exit 1; fi
# get array length
N=${#POS_NAMES[@]}

# start one job per position
# using a job array is not convenient since it requires to pass indices and flip flags arrays as arguments
for (( I=0; I<N; I++ )); do
  POS_NAME=${POS_NAMES[$I]}
  PREPROC_DIR=$(printf $PREPROC_DIR_TPL $POS_NAME)
  CROP_ROI_PATH=$(printf $CROP_ROI_PATH_TPL $POS_NAME)
  ROTATION=${ROTATIONS[$I]}

  BASENAME=$(basename "$PREPROC_DIR")
  SCRIPT=$PREPROC_DIR/logs/slurm_${BASENAME}.sh           # path to custom bash script
  LOG=$PREPROC_DIR/logs/slurm_${BASENAME}.log             # path to redirect stdout (qsub command and job ID)
  S_OUT=$PREPROC_DIR/logs/slurm_${BASENAME}_out.log  # path to redirect qsub stdout
  S_ERR=$PREPROC_DIR/logs/slurm_${BASENAME}_err.log  # path to redirect qsub stderr
  
  mkdir -p $PREPROC_DIR # -p: no error if existing, make parent directories as needed
  mkdir -p $PREPROC_DIR/logs
#   CMD="/scicore/home/nimwegen/GROUP/MM_Analysis/mm_pre_bc2.sh $RAW_FILE $PREPROC_DIR $ROTATION $CHANNELS_ORDER"
  CMD_STR="$MMPRE_HOME/mm_pre_slurm.sh \
  -i \"$RAW_PATH\" \
  -o $PREPROC_DIR"
  if [ -n "$POS_NAME" ]; then CMD_STR="$CMD_STR -p $POS_NAME"; fi # append optional argument
  if ! (( $ROTATION == 0 )); then CMD_STR="$CMD_STR -r $ROTATION"; fi # append optional argument
  if [ -n "$CAMERA_ROI_PATH" ]; then CMD_STR="$CMD_STR -j $CAMERA_ROI_PATH"; fi # append optional argument
  if [ -n "$CROP_ROI_PATH" ]; then CMD_STR="$CMD_STR -k $CROP_ROI_PATH"; fi # append optional argument
  if [ -n "$DARK_PATH" ]; then CMD_STR="$CMD_STR -d $DARK_PATH"; fi # append optional argument
  if [ -n "$GAIN_PATH" ]; then CMD_STR="$CMD_STR -g $GAIN_PATH"; fi # append optional argument
  if [ -n "$HOTPIXS_PATH" ]; then CMD_STR="$CMD_STR -h $HOTPIXS_PATH"; fi # append optional argument
  if [ -n "$FLATFIELD_PATH" ]; then CMD_STR="$CMD_STR -ff $FLATFIELD_PATH"; fi # append optional argument
  if [ -n "$CHANNELS_ORDER" ]; then CMD_STR="$CMD_STR -c $CHANNELS_ORDER"; fi # append optional argument
  if [ -n "$FIJI_MACRO" ]; then CMD_STR="$CMD_STR -m $FIJI_MACRO"; fi # append optional argument
  if [ -n "$TMIN" ]; then CMD_STR="$CMD_STR -tmin $TMIN"; fi # append optional argument
  if [ -n "$TMAX" ]; then CMD_STR="$CMD_STR -tmax $TMAX"; fi # append optional argument
  
  # use single quote to prevent variable evaluation
  CMD_SCRIPT="#!/bin/bash \n\n\
#SBATCH -t 1-00:00:00 \n\
#SBATCH --qos=1day \n\
#SBATCH --nodes=1 \n\
#SBATCH --ntasks=2 \n\
#SBATCH --mem-per-cpu=32G \n\
#SBATCH --export=MODULEPATH=$MODULEPATH \n\
#SBATCH -o $S_OUT \n\
#SBATCH -e $S_ERR \n\n\
$CMD_STR \n"
  CMD_SBATCH="sbatch $SCRIPT"

  echo CMD_SCIPT:
  echo $CMD_SCRIPT
  echo SCRIPT:
  echo $SCRIPT
  
  printf "$CMD_SCRIPT" > $SCRIPT
  
  [ -f "$S_OUT" ] && rm $S_OUT # delete if exists
  [ -f "$S_ERR" ] && rm $S_ERR
  printf "$CMD_SBATCH \n" > $LOG
  
  echo LOG:
  echo $LOG

  $CMD_SBATCH >> $LOG
done
echo "Preprocessing queued... (use squeue to check the current status)"
wait
}
