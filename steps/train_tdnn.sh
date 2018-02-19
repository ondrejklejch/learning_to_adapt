#!/bin/bash

. path.sh
. cmd.sh

# TODO: Use locking script to obtain GPU
export CUDA_VISIBLE_DEVICES=2
export TF_CPP_MIN_LOG_LEVEL=2

adapt_ali="exp/kaldi_tdnn/align_dev2010_hires_concatenated/"
test_ali="exp/kaldi_tdnn/align_dev2010_hires_concatenated/"
data="data/dev2010_hires_concatenated/"
model="exp/kaldi_tdnn/final.mdl"

keras_model="exp/model_tdnn/dnn.nnet.h5"
feats="scp:$data/feats.scp"
utt2spk=$data/utt2spk
adapt_pdfs="ark:ali-to-pdf $model ark:'gunzip -c $adapt_ali/ali.*.gz |' ark,t:- |"
test_pdfs="ark:ali-to-pdf $model ark:'gunzip -c $test_ali/ali.*.gz |' ark,t:- |"
output="exp/meta_tdnn_supervised.h5"

frame_subsampling_factor=`cat exp/kaldi_tdnn/frame_subsampling_factor`
context_opts=`cat exp/model_tdnn/context_opts`


python2.7 steps/train.py $keras_model "$feats" $utt2spk "$adapt_pdfs" "$test_pdfs" ALL $output "sequences" $frame_subsampling_factor $context_opts
