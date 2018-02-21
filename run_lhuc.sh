#!/bin/bash

. cmd.sh
. path.sh

nj=4
dir=exp/lhuc_256_unsupervised
configs=$dir/configs
mkdir -p $configs
rm -rf $configs/*
for lr in 0.4 0.5 0.6; do
    for epochs in 1 3 5; do
        name="lhuc_${lr}_${epochs}"
        echo "{\"lr\": $lr, \"epochs\": $epochs}" > $configs/$name.json
        echo -e "${name}\t${configs}/${name}.json" >> $configs/experiments.scp
    done
done

mkdir -p $configs/split${nj}
for job in `seq 1 $nj`; do
    utils/split_scp.pl -j $nj $((job-1)) $configs/experiments.scp $configs/split${nj}/$job.scp
done

#for dataset in dev2010 tst2010 tst2011; do
for dataset in tst2011; do
    data=data/${dataset}
    pdfs=exp/dnn_256-7-small_softmax-dbn_dnn/decode_${dataset}/ali/
    frames=1000
    model=exp/model_with_lhuc/
    graph=exp/model/graph_TED-312MW.3gm.p07/
    decode_dir=$dir/decode_${dataset}

    ln -s `pwd`/$model/final.mdl $decode_dir/final.mdl
    ln -s `pwd`/data/lang $decode_dir/lang
    ln -s `pwd`/local/scoring/stms/ted.${dataset}.en-fr.en.norm.stm $decode_dir/stm

    echo "Decoding: $dataset"
    steps/create_splits_by_spk.sh $data
    $cmd JOB=1:$nj $decode_dir/log/experiments.JOB.log \
        steps/run_experiments.sh LHUC $configs/split${nj}/JOB.scp $data $pdfs $frames $model $graph $decode_dir

    echo
    echo "Best result $dataset"
    grep "Percent Total Error" $decode_dir/*/*/best_wer | sed 's/:.*= */ /;s/%.*/%/;' | sort -n -k2,2 | head -n 1
done;
