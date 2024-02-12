#!/usr/bin/env bash

# Pass along all ENV variables as arguments to my Python script
#CUDA_VISIBLE_DEVICES=$gpu_idx python ../../src/image_level/mixmatch.py \
#    --train_kimg $train_kimg \
#    --train_dir $train_dir \
#    --task_name $task_name \
#    --train_labeled_files $train_labeled_files \
#    --train_unlabeled_files $train_unlabeled_files \
#    --valid_files $valid_files \
#    --test_files $test_files \
#    --class_weights $class_weights \
#    --lr $lr \
#    --wd $wd \
#    --w_match $w_match\
#    --warmup_kimg $warmup_kimg\
#    --warmup_delay $warmup_delay\
#    --mixmode $mixmode \
#    --report_type $report_type\

CUDA_VISIBLE_DEVICES=2 python "/home/baharkhd/ssl-for-echocardiograms-miccai2024/src/image_level/mixmatch.py" \
    --train_kimg 2000 \
    --train_dir "/home/baharkhd/ssl-for-echocardiograms-miccai2024/experiments/table5/MixMatch/fold0" \
    --task_name "DiagnosisClassification" \
    --train_labeled_files 'train_DIAGNOSIS.tfrecord' \
    --train_unlabeled_files 'unlabeled_DIAGNOSIS.tfrecord' \
    --valid_files 'val_DIAGNOSIS.tfrecord' \
    --test_files 'test_DIAGNOSIS.tfrecord' \
    --class_weights "0.3385,0.3292,0.3323" \
    --lr 0.0007 \
    --wd 0.0002 \
    --w_match 75.0\
    --warmup_kimg 1024\
    --warmup_delay 500\
    --mixmode 'xxy.yxy' \
    --report_type "EMA_BalancedAccuracy"\


    if [[ -z $1 ]]; then
    ACTION_NAME='list'
else
    ACTION_NAME=$1
fi

#export gpu_idx=4 
#export ML_DATA="../../ML_DATA/TMED-18-18/fold0/"
#export PYTHONPATH=$PYTHONPATH:.
#export train_kimg=2000
#export class_weights="0.3385,0.3292,0.3323"
#export lr=0.0007
#export wd=0.0002
#export w_match=75.0
#export warmup_delay=500
#export warmup_kimg=1024
#export train_dir="../../experiments/table5/MixMatch/fold0"
#export task_name="DiagnosisClassification"
#export report_type="EMA_BalancedAccuracy"
#export train_labeled_files=
#export train_unlabeled_files='unlabeled_DIAGNOSIS.tfrecord'
#export valid_files='val_DIAGNOSIS.tfrecord'
#export test_files='test_DIAGNOSIS.tfrecord'
#export mixmode='xxy.yxy'