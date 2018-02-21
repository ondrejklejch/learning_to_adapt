#!/bin/bash

. cmd.sh
. path.sh

nj=1
dir="exp/meta_large_lhuc_lr_per_parameter/"
configs=$dir/configs
mkdir -p $configs
rm -rf $configs/*

echo "{\"model\": \"exp/meta_large_lhuc_lr_per_parameter/meta_large_lr_per_parameter.h5\", \"epochs\": 5}" > $configs/meta.json
echo -e "meta\t${configs}/meta.json" > $configs/experiments.scp

mkdir -p $configs/split${nj}
for job in `seq 1 $nj`; do
    utils/split_scp.pl -j $nj $((job-1)) $configs/experiments.scp $configs/split${nj}/$job.scp
done

for dataset in dev2010 tst2010 tst2011; do
    data=data/${dataset}
    pdfs="exp/dnn5_fbank/align_${dataset}/"
    frames=1000
    model="exp/dnn5_fbank_model_with_lhuc/"
    graph="exp/dnn5_fbank/graph_TED-312MW.3gm.p07/"
    decode_dir=$dir/decode_${dataset}

    steps/create_splits_by_spk.sh $data

    $cmd JOB=1:$nj $decode_dir/log/experiments.JOB.log \
        steps/run_experiments.sh META $configs/split${nj}/JOB.scp $data $pdfs $frames $model $graph $decode_dir

    ln -s `pwd`/$model/final.mdl $decode_dir/final.mdl
    for experiment in `ls -1 $decode_dir | grep -v log | grep -v final.mdl`; do
        time local/score_ted.sh --stm local/scoring/stms/ted.${dataset}.en-fr.en.norm.stm data/${dataset} data/lang/ $decode_dir/$experiment
    done

    echo "Best result"
    grep "Percent Total Error" $decode_dir/*/*/best_wer | sed 's/:.*= */ /;s/%.*/%/;' | sort -n -k2,2 | head -n 1
done
