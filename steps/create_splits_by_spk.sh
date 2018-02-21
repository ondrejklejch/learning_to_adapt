#!/bin/bash

data=$1

cut -f 1 -d ' ' $data/spk2utt > $data/spks_list
num_spks=`cat $data/spks_list | wc -l`

mkdir -p $data/spks_split
for i in `seq $num_spks`; do
    split_dir=$data/spks_split/$i/

    mkdir -p $split_dir
    sed -n "${i}p" $data/spks_list > $split_dir/spks_list
    utils/subset_data_dir.sh --spk-list $split_dir/spks_list $data $split_dir || exit 1;
done
