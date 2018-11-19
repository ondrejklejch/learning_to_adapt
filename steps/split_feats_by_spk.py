import sys
from collections import defaultdict


if __name__ == '__main__':
    feats_rspecifier = sys.argv[1]
    train_dir = sys.argv[2]
    val_dir = sys.argv[3]
    num_valid_spks = int(sys.argv[4])

    feats = defaultdict(list)
    with open(feats_rspecifier, 'r') as f:
        for line in f:
            line = line.strip()
            utt, _ = line.split(None, 1)
            spk = "_".join(utt.split('_')[:-1])

            feats[spk].append(line)

    for i, all_feats in enumerate(feats.values()[:-num_valid_spks]):
        with open('%s/feats_%.4d.scp' % (train_dir, i + 1), 'w') as f:
            for line in all_feats:
                print >> f, line

    for i, all_feats in enumerate(feats.values()[-num_valid_spks:]):
        with open('%s/feats_%.4d.scp' % (val_dir, i + 1), 'w') as f:
            for line in all_feats:
                print >> f, line
