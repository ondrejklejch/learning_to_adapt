#!/bin/bash

. path.sh
. cmd.sh

# TODO: Use locking script to obtain GPU
export CUDA_VISIBLE_DEVICES=1
export TF_CPP_MIN_LOG_LEVEL=2

# Obtain alignments for adaptation
data="data/train_cleaned_sp_hires_comb/"
ali="exp/tri3_cleaned_ali_train_cleaned_sp_comb/"
model="$ali/final.mdl"

if [ ! -f $ali/ali.txt ]; then
    ali-to-pdf $model "ark:gunzip -c $ali/ali.*.gz |" ark,t:$ali/ali.txt
fi

adapt_pdfs="ark:$ali/ali.txt"
test_pdfs="ark:$ali/ali.txt"

frame_subsampling_factor=1
context_opts="-16 12"

# Prepare training data splits
if [ ! -d $data/keras_meta_train_split ]; then
    mkdir -p $data/keras_meta_{train,val}_split
    python2.7 steps/split_feats_by_spk.py $data/feats.scp $data/keras_meta_train_split $data/keras_meta_val_split 10
fi

model_dir="exp/maml_renorm_350/"
mkdir -p $model_dir/
python2.7 steps/meta/train_sat_maml.py $data/keras_meta_train_split $data/keras_meta_val_split "$adapt_pdfs" "$test_pdfs" LHUC $model_dir $frame_subsampling_factor $context_opts
