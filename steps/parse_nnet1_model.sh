#!/bin/bash

src_dir=$1
out_dir=$2

mkdir -p $out_dir

nnet-copy --binary=false $src_dir/final.nnet $out_dir/final.txt || exit 1
cp -r $src_dir/graph_* $out_dir
cp $src_dir/{tree,final.mdl,ali_train_pdf.counts,final.feature_transform} $out_dir

python steps/parse_nnet1_model.py $out_dir
