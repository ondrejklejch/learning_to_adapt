#!/bin/bash

experiments=$1
data=$2
pdfs=$3
frames=$4
model=$5
graph=$6
decode_dir=$7

while read line; do
    name=`echo $line | awk '{print $1}'`
    path=`echo $line | awk '{print $2}'`

    echo $name
    echo $path
    bash steps/adapt_and_decode.sh LHUC $path $data $pdfs $frames $model $graph $decode_dir/$name
done < $experiments
