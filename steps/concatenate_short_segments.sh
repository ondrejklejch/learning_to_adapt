#!/bin/bash

src=$1
dest=$2

rm -rf $dest
mkdir -p $dest

python steps/concatenate_short_segments.py $src $dest 0.5 10.0
utils/utt2spk_to_spk2utt.pl $dest/utt2spk > $dest/spk2utt
cp $src/{wav.scp,reco2file_and_channel} $dest

utils/validate_data_dir.sh --no-feats $dest

