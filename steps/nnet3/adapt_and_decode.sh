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

for f in $graph_dir/HCLG.fst $data/feats.scp; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

# Prepare alignments
mkdir -p $decode_dir
ali-to-pdf $model_dir/final.mdl ark:"gunzip -c $adaptation_ali/ali.*.gz |" ark,t:$decode_dir/pdfs

cmvn_opts=`cat $model_dir/cmvn_opts`
delta_opts=`cat $model_dir/delta_opts`
splice_opts=`cat $model_dir/splice_opts`
context_opts=`cat $model_dir/context_opts`
frame_subsampling_factor=`cat $model_dir/frame_subsampling_factor`

num_spks=`cat $data/spk2utt | wc -l`
sdata=$data/split$num_spks
adaptation_pdfs="ark:$decode_dir/pdfs"
feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |"
feats="$feats add-deltas $delta_opts ark:- ark:- |"
feats="$feats splice-feats $splice_opts ark:- ark:- |"
feats="$feats python steps/nnet3/adapt_and_decode.py $method $config $adaptation_pdfs $adaptation_frames $model_dir/dnn.nnet.h5 $model_dir/pdf_counts $frame_subsampling_factor $context_opts |"
feats="$feats grep -v 'import dot_parser' |"
decode_opts="--max-active=$max_active --beam=$beam --lattice-beam=$latbeam --acoustic-scale=$acwt --allow-partial=true"
lat_wspecifier="ark:| gzip -c > $decode_dir/lat.JOB.gz"

$cmd JOB=1:$num_spks $decode_dir/log/decode.JOB.log \
  latgen-faster-mapped $decode_opts --word-symbol-table=$graph_dir/words.txt $model_dir/final.mdl $graph_dir/HCLG.fst "$feats" "$lat_wspecifier"

bash local/score.sh $data $graph_dir $decode_dir
grep "Percent Total Error" $decode_dir/score_*/*.dtl | sort -k 5,5nr | tail -n 1
