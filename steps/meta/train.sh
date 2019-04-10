#!/bin/bash

. path.sh
. cmd.sh

# TODO: Use locking script to obtain GPU
export CUDA_VISIBLE_DEVICES=1
export TF_CPP_MIN_LOG_LEVEL=2


dset="dev_iwslt"
adaptation_type="sup"
adapted_weights="all"
model_dir="exp/tdnn_am_600_batchnorm/"

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

for mode in lr lr_per_step lr_per_layer lr_per_layer_per_step; do
    for num_frames in 1000 3000 6000; do
        echo "Training $adaptation_type $adapted_weights $mode with $num_frames for $model_dir"

        meta_dir=$model_dir/meta_${adaptation_type}_${adapted_weights}_${mode}_${num_frames}
        mkdir -p $meta_dir
        python2.7 steps/meta/train.py $model_dir/model.best.h5 $data/keras_meta_train_split $data/keras_meta_val_split "$adapt_pdfs" "$test_pdfs" $adapted_weights $meta_dir $frame_subsampling_factor $context_opts $num_frames $mode
    done
done
