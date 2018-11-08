#!/bin/bash

. path.sh
. cmd.sh

export CUDA_VISIBLE_DEVICES=
export TF_CPP_MIN_LOG_LEVEL=2

dir=$1
python2.7 steps/nnet3/convert_sat_model.py $dir/model.best.h5 $dir/model.best.normal.h5
