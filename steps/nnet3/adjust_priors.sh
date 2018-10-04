#!/bin/bash

. cmd.sh
. path.sh

cmd=run.pl
data=$1
model_dir=$2

for f in $data/feats.scp; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

cmvn_opts=`cat $model_dir/cmvn_opts`
delta_opts=`cat $model_dir/delta_opts`
splice_opts=`cat $model_dir/splice_opts`
context_opts=`cat $model_dir/context_opts`
frame_subsampling_factor=`cat $model_dir/frame_subsampling_factor`

# Split data by speaker
num_spks=`cat $data/spk2utt | wc -l`
utils/split_data.sh $data $num_spks

sdata=$data/split${num_spks}
feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |"
feats="$feats add-deltas $delta_opts ark:- ark:- |"
feats="$feats splice-feats $splice_opts ark:- ark:- |"
feats="$feats python steps/nnet3/nnet3_forward.py $model_dir/dnn.nnet.h5 $model_dir/pdf_counts $frame_subsampling_factor $context_opts True |"
feats="$feats grep -v 'import dot_parser' |"

rm $model_dir/pdf_counts
$cmd JOB=1:$num_spks $model_dir/log/adjust_priors.JOB.log \
    matrix-sum-rows "$feats" ark:$model_dir/posteriors.JOB.vec

vector-sum "ark:cat $model_dir/posteriors.*.vec |" $model_dir/pdf_counts
