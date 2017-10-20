#!/bin/bash

. cmd.sh
. path.sh

nj=1
dir=exp/meta
configs=$dir/configs
mkdir -p $configs

echo "{\"model\": \"exp/meta/meta.h5\"}" > $configs/meta.json
echo -e "meta\t${configs}/meta.json" >> $configs/experiments.scp

mkdir -p $configs/split${nj}
for job in `seq 1 $nj`; do
    utils/split_scp.pl -j $nj $((job-1)) $configs/experiments.scp $configs/split${nj}/$job.scp
done

data=data/dev2010
pdfs=exp/dnn_256-7-small_softmax-dbn_dnn/align_dev2010/
frames=1000
model=exp/model_with_lhuc/
graph=exp/model/graph_TED-312MW.3gm.p07/
decode_dir=$dir/decode_dev2010

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

$cmd JOB=1:$nj $decode_dir/log/experiments.JOB.log \
    steps/run_experiments.sh META $configs/split${nj}/JOB.scp $data $pdfs $frames $model $graph $decode_dir

ln -s `pwd`/$model/final.mdl $decode_dir/final.mdl
for experiment in `ls -1 $decode_dir | grep -v log`; do
    time local/score_ted.sh --stm local/scoring/stms/ted.dev2010.en-fr.en.norm.stm data/dev2010 data/lang/ $decode_dir/$experiment
done

echo "Best result"
grep "Percent Total Error" $decode_dir/*/*/best_wer | sed 's/:.*= */ /;s/%.*/%/;' | sort -n -k2,2 | head -n 1
