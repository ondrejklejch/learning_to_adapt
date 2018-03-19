#!/bin/bash

method=$1
experiments=$2
data=$3
pdfs=$4
frames=$5
model=$6
graph=$7
decode_dir=$8

while read line; do
    name=`echo $line | awk '{print $1}'`
    path=`echo $line | awk '{print $2}'`

    echo $name
    echo $path
    bash steps/adapt_and_decode.sh $method $path $data $pdfs $frames $model $graph $decode_dir/$name
    bash local/score_ted.sh --stm $decode_dir/stm $data $graph $decode_dir/$name
done < $experiments
