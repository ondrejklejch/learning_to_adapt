#!/bin/bash

. path.sh
. cmd.sh

# TODO: Use locking script to obtain GPU
export CUDA_VISIBLE_DEVICES=3
export TF_CPP_MIN_LOG_LEVEL=2

ali="exp/dnn_256-7-small_softmax-dbn_dnn/align_dev2010/"
data="data/dev2010/"
model="exp/model/final.mdl"

keras_model="exp/model/dnn.nnet.h5"
feats="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp scp:$data/feats.scp ark:- 2> /dev/null |"
feats="$feats add-deltas ark:- ark:- 2> /dev/null |"
feats="$feats splice-feats --left-context=3 --right-context=3 ark:- ark:- 2> /dev/null |"
utt2spk=$data/utt2spk
pdfs="ark:ali-to-pdf $model ark:'gunzip -c $ali/ali.*.gz |' ark,t:- |"
output="exp/meta.h5"

python2.7 train.py $keras_model "$feats" $utt2spk "$pdfs" "$pdfs" $output
