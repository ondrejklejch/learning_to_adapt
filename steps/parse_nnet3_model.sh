#!/bin/bash

. cmd.sh
. path.sh

src_dir=$1
out_dir=$2
with_lhuc_layers=$3

mkdir -p $out_dir

frame_subsampling=1
if [ -f $src_dir/frame_subsampling_factor ]; then
  frame_subsampling=$(cat $src_dir/frame_subsampling_factor)
fi

echo $frame_subsampling > $out_dir/frame_subsampling_factor
cp -r $src_dir/{final.mdl,graph,cmvn_opts,pdf_counts} $out_dir
nnet3-copy --binary=false --prepare-for-test=true $out_dir/final.mdl $out_dir/final.txt
python steps/parse_nnet3_model.py $out_dir $frame_subsampling $with_lhuc_layers
