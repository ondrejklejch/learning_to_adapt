#!/bin/bash

. cmd.sh
. path.sh

cmd=run.pl
max_active=7000 # max-active
beam=15.0
latbeam=7.0
acwt=0.1

method=$1
config=$2
data=$3
adaptation_ali=$4
adaptation_frames=$5
model_dir=$6
graph_dir=$7
decode_dir=$8

for f in $graph_dir/HCLG.fst $data/feats.scp $model_dir/tree; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

# Prepare alignments
mkdir -p $decode_dir
ali-to-pdf $model_dir/final.mdl ark:"gunzip -c $adaptation_ali/ali.*.gz |" ark,t:$decode_dir/pdfs

sdata=$data/spks_split
adaptation_pdfs="ark:$decode_dir/pdfs"
feats="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |"
$add_deltas && feats="$feats add-deltas ark:- ark:- |"
$splice_feats && feats="$feats splice-feats --left-context=3 --right-context=3 ark:- ark:- |"
feats="$feats python steps/adapt_and_decode.py $method $config $adaptation_pdfs $adaptation_frames $model_dir/dnn.nnet.h5 $model_dir/dnn.priors.csv |"
feats="$feats grep -v 'import dot_parser' |"

num_spks=`cat $data/spks_list | wc -l`
$cmd JOB=1:$num_spks $decode_dir/log/decode.JOB.log \
  latgen-faster-mapped --max-active=$max_active --beam=$beam --lattice-beam=$latbeam --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$graph_dir/words.txt $model_dir/final.mdl $graph_dir/HCLG.fst "$feats" "ark:|gzip -c > $decode_dir/lat.JOB.gz"

