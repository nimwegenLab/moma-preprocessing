DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

ml purge
ml Python/3.5.2-goolf-1.7.20
source venv-testenv/bin/activate

pip install ipdb
pip install pudb
pip install exifread

cd colicycle
pip install . --upgrade

mkdir -p $DIR/testoutput

LOGFILE=$DIR/output.log

INPUT_DIR=$DIR/20150710_mmtest_2ch_rawstacks
OUTPUT_DIR=$DIR/testoutput
python $DIR/preproc_script.py  $INPUT_DIR $OUTPUT_DIR 1 3
# >> output.log
#OUTPUT_DIR=/scicore/home/nimwegen/GROUP/MM_Data/Guillaume/20180723_GW277_glucose37/20180723_GW277_glucose37_1/
#python "$DIR/preproc_script.py" "$INPUT_DIR" "$OUTPUT_DIR" "0" "375"
# >> output.log


#python preproc_script.py "/scicore/home/nimwegen/GROUP/MM_Data/Guillaume/20180723_GW277_glucose37/20180723_GW277_glucose37_1/" "/scicore/home/nimwegen/micuby52/Documents/DeepLearningMoM_OUTPUT" "0" "375" 

echo ========== $(date) ========== >> output.log
