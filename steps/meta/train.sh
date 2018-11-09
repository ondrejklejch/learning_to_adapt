#!/bin/bash

. path.sh
. cmd.sh

# TODO: Use locking script to obtain GPU
export CUDA_VISIBLE_DEVICES=2
export TF_CPP_MIN_LOG_LEVEL=2


dset="dev2010-2012"
adaptation_type="sup"
model_dir="exp/tdnn_am_850_renorm/"

# Obtain alignments for adaptation
if [ $adaptation_type == "sup" ]; then
    adapt_ali="exp/kaldi_tdnn/align_$dset/"
else
    adapt_ali="exp/kaldi_tdnn/decode_$dset/ali/"
fi
test_ali="exp/kaldi_tdnn/align_$dset/"
model="exp/kaldi_tdnn/final.mdl"
adapt_pdfs="ark:ali-to-pdf $model ark:'gunzip -c $adapt_ali/ali.*.gz |' ark,t:- |"
test_pdfs="ark:ali-to-pdf $model ark:'gunzip -c $test_ali/ali.*.gz |' ark,t:- |"

frame_subsampling_factor=1
context_opts="-16 12"

# Prepare training data splits
data="data/${dset}_hires/"
if [ ! -d $data/keras_meta_train_split ]; then
    mkdir -p $data/keras_meta_{train,val}_split
    python2.7 steps/split_feats_by_spk.py $data/feats.scp $data/keras_meta_train_split $data/keras_meta_val_split 5
fi

mkdir -p $model_dir/meta_$adaptation_type
python2.7 steps/meta/train.py $model_dir/model.best.h5 $data/keras_meta_train_split $data/keras_meta_val_split "$adapt_pdfs" "$test_pdfs" ALL $model_dir/meta_$adaptation_type $frame_subsampling_factor $context_opts
