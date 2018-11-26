#!/bin/bash

. path.sh
. cmd.sh

export CUDA_VISIBLE_DEVICES=
export TF_CPP_MIN_LOG_LEVEL=2

dir=$1
python2.7 steps/meta/convert_maml_model.py $dir/meta.best.h5 $dir/meta.best.h5 $dir/model.best.h5
