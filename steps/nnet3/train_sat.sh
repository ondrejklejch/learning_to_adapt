#!/bin/bash

. path.sh
. cmd.sh

# TODO: Use locking script to obtain GPU
export CUDA_VISIBLE_DEVICES=2
export TF_CPP_MIN_LOG_LEVEL=2

ali="exp/tri3_cleaned_ali_train_cleaned_sp_comb/"

data="data/train_cleaned_sp_hires_comb/"
utt2spk=$data/utt2spk
pdfs="ark:ali-to-pdf $ali/final.mdl ark:'gunzip -c $ali/ali.*.gz |' ark,t:- |"
left_context=-16
right_context=12
lda="lda.txt"
output="exp/tdnn_am_850_sat_batchnorm/"

num_splits=1000
if [ ! -d $data/keras_train_split ]; then
    mkdir $data/keras_train_split

    sort -R $data/feats.scp > $data/keras_train_split/all_feats.scp
    tail -n +301 $data/keras_train_split/all_feats.scp > $data/keras_train_split/feats.scp
    split --additional-suffix .scp --numeric-suffixes -n l/$num_splits -a 4 $data/keras_train_split/feats.scp $data/keras_train_split/feats_

    mkdir $data/keras_val_split
    head -n 300 $data/keras_train_split/all_feats.scp > $data/keras_val_split/feats.scp
    split --additional-suffix .scp --numeric-suffixes -n l/10 -a 4 $data/keras_val_split/feats.scp $data/keras_val_split/feats_

    rm $data/keras_train_split/all_feats.scp
fi


mkdir -p $output
python2.7 steps/nnet3/train_sat.py $data/keras_train_split $data/keras_val_split $utt2spk "$pdfs" $left_context $right_context $lda $output
