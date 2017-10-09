#!/bin/bash

. cmd.sh
. path.sh

cmd=run.pl
max_active=7000 # max-active
beam=15.0
latbeam=7.0
acwt=0.1

data=$1
adaptation_ali=$2
adaptation_frames=$3
model_dir=$4
graph_dir=$5
decode_dir=$6

for f in $graph_dir/HCLG.fst $data/feats.scp $model_dir/tree; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

# Create splits by speakers
cut -f 1 -d ' ' $data/spk2utt > $data/spks_list
num_spks=`cat $data/spks_list | wc -l`

mkdir -p $data/spks_split
for i in `seq $num_spks`; do
    dir=$data/spks_split/$i/

    mkdir -p $dir
    sed -n "${i}p" $data/spks_list > $dir/spks_list
    utils/subset_data_dir.sh --spk-list $dir/spks_list $data $dir || exit 1;
done

# Prepare alignments
ali-to-pdf $model_dir/final.mdl ark:"gunzip -c $adaptation_ali/ali.*.gz |" ark,t:$decode_dir/pdfs

sdata=$data/spks_split
adaptation_pdfs="ark:$decode_dir/pdfs"
feats="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |"
$add_deltas && feats="$feats add-deltas ark:- ark:- |"
$splice_feats && feats="$feats splice-feats --left-context=3 --right-context=3 ark:- ark:- |"
feats="$feats python steps/adapt_and_decode.py '$feats' $adaptation_pdfs $adaptation_frames $model_dir/dnn.nnet.h5 $model_dir/dnn.priors.csv |"
feats="$feats grep -v 'import dot_parser' |"

$cmd JOB=1:$num_spks $decode_dir/log/decode.JOB.log \
  latgen-faster-mapped --max-active=$max_active --beam=$beam --lattice-beam=$latbeam --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$graph_dir/words.txt $model_dir/final.mdl $graph_dir/HCLG.fst "$feats" "ark:|gzip -c > $decode_dir/lat.JOB.gz"

