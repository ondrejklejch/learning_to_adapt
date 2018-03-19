#!/bin/bash

. path.sh

src_dir=$1
out_dir=$2

mkdir -p $out_dir

nnet-copy --binary=false $src_dir/final.nnet $out_dir/final.txt || exit 1
nnet-info $src_dir/final.feature_transform | grep frame_offsets | awk '{printf("--left-context=%d --right-context=%d", ($3 < 0) ? -$3 : $3, $(NF-1))}' > $out_dir/splice_opts

echo 1 > $out_dir/frame_subsampling_factor
echo 0 0 > $out_dir/context_opts

cp -r $src_dir/graph_* $out_dir
cp $src_dir/{tree,final.mdl,final.feature_transform,cmvn_opts,delta_opts} $out_dir
cp $src_dir/ali_train_pdf.counts $out_dir/pdf_counts

python steps/parse_nnet1_model.py $out_dir
