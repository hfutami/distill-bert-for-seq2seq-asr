#!/usr/bin/env bash

# https://github.com/kaldi-asr/kaldi/blob/master/egs/csj/s5/run.sh

set -e # exit on error

#: << '#SKIP'

use_dev=true # Use the first 4k sentences from training data as dev set. (39 speakers.)

CSJDATATOP=$1
#CSJDATATOP=/db/laputa1/data/processed/public/CSJ ## CSJ database top directory.
CSJVER=dvd  ## Set your CSJ format (dvd or usb).
            ## Usage    :
            ## Case DVD : We assume CSJ DVDs are copied in this directory with the names dvd1, dvd2,...,dvd17.
            ##            Neccesary directory is dvd3 - dvd17.
            ##            e.g. $ ls $CSJDATATOP(DVD) => 00README.txt dvd1 dvd2 ... dvd17
            ##
            ## Case USB : Neccesary directory is MORPH/SDB and WAV
            ##            e.g. $ ls $CSJDATATOP(USB) => 00README.txt DOC MORPH ... WAV fileList.csv
            ## Case merl :MERL setup. Neccesary directory is WAV and sdb

if [ ! -e data/csj-data/.done_make_all ]; then
 echo "CSJ transcription file does not exist"
 #local/csj_make_trans/csj_autorun.sh <RESOUCE_DIR> <MAKING_PLACE(no change)> || exit 1;
 local/csj_make_trans/csj_autorun.sh $CSJDATATOP data/csj-data $CSJVER
fi
wait

[ ! -e data/csj-data/.done_make_all ]\
    && echo "Not finished processing CSJ data" && exit 1;

# Prepare Corpus of Spontaneous Japanese (CSJ) data.
# Processing CSJ data to KALDI format based on switchboard recipe.
# local/csj_data_prep.sh <SPEECH_and_TRANSCRIPTION_DATA_DIRECTORY> [ <mode_number> ]
# mode_number can be 0, 1, 2, 3 (0=default using "Academic lecture" and "other" data, 
#                                1=using "Academic lecture" data, 
#                                2=using All data except for "dialog" data, 3=using All data )
local/csj_data_prep.sh data/csj-data
# local/csj_data_prep.sh data/csj-data 1
# local/csj_data_prep.sh data/csj-data 2
# local/csj_data_prep.sh data/csj-data 3

# Data preparation and formatting for evaluation set.
# CSJ has 3 types of evaluation data
#local/csj_eval_data_prep.sh <SPEECH_and_TRANSCRIPTION_DATA_DIRECTORY_ABOUT_EVALUATION_DATA> <EVAL_NUM>
for eval_num in eval1 eval2 eval3 ; do
    local/csj_eval_data_prep.sh data/csj-data/eval $eval_num
done

echo "Now make MFCC features"

# Now make MFCC features.
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
mfccdir=mfcc

for x in train eval1 eval2 eval3; do
  steps/make_mfcc.sh --nj 50 --cmd "$train_cmd" \
    data/$x exp/make_mfcc/$x $mfccdir
  steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir
  utils/fix_data_dir.sh data/$x
done

echo "Finish creating MFCCs"
