#!/bin/bash

. path.sh
. cmd.sh

# TODO: Use locking script to obtain GPU
export CUDA_VISIBLE_DEVICES=2
export TF_CPP_MIN_LOG_LEVEL=2

ali="exp/kaldi_tdnn_subset/align_dev2010-2012/"

data="data/dev2010-2012_hires/"
feats="scp:$data/feats.scp"
utt2spk=$data/utt2spk
pdfs="ark:ali-to-pdf $ali/final.mdl ark:'gunzip -c $ali/ali.*.gz |' ark,t:- |"
left_context=-16
right_context=12
output="exp/tdnn_am_batch256_20181030/"


mkdir -p $output
python2.7 steps/nnet3/nnet3_average.py "$feats" $utt2spk "$pdfs" $left_context $right_context $output
