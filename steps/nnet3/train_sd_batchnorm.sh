#!/bin/bash

. path.sh
. cmd.sh

# TODO: Use locking script to obtain GPU
export CUDA_VISIBLE_DEVICES=3
export TF_CPP_MIN_LOG_LEVEL=2

ali="exp/tri3_cleaned_ali_train_cleaned_sp_comb/"

data="data/train_cleaned_sp_hires_comb/"
utt2spk=$data/utt2spk
pdfs="ark:$ali/ali.txt"
left_context=-16
right_context=12
lda="lda.txt"
output="exp/tdnn_am_850_sd_batchnorm_2/"

if [ ! -f $ali/ali.txt ]; then
  ali-to-pdf $ali/final.mdl "ark:gunzip -c $ali/ali.*.gz |" ark,t:$ali/ali.txt
fi

# Prepare training data splits
if [ ! -d $data/keras_sd_batchnorm_train_split ]; then
    mkdir -p $data/keras_sd_batchnorm_{train,val}_split
    python2.7 steps/split_feats_by_utts.py $data/feats.scp $data/keras_sd_batchnorm_train_split $data/keras_sd_batchnorm_val_split 3
fi

mkdir -p $output
python2.7 steps/nnet3/train_sd_batchnorm.py $data/keras_sd_batchnorm_train_split $data/keras_sd_batchnorm_val_split $utt2spk "$pdfs" $left_context $right_context $lda $output
