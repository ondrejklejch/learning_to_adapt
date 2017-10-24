import numpy as np
import kaldi_io
import collections


def load_data(params, feats, utt2spk, adapt_pdfs, test_pdfs, num_frames=1000, steps=1):
    utt_to_adapt_pdfs = load_utt_to_pdfs(adapt_pdfs)
    utt_to_test_pdfs = load_utt_to_pdfs(test_pdfs)
    utt_to_spk = load_utt_to_spk(utt2spk)
    feats_reader = kaldi_io.SequentialBaseFloatMatrixReader(feats)

    feats = collections.defaultdict(list)
    adapt_pdfs = collections.defaultdict(list)
    test_pdfs = collections.defaultdict(list)

    for (utt, utt_feats) in feats_reader:
        if utt not in utt_to_adapt_pdfs or utt not in utt_to_test_pdfs:
            continue

        spk = utt_to_spk[utt]
        utt_adapt_pdfs = utt_to_adapt_pdfs[utt]
        utt_test_pdfs = utt_to_test_pdfs[utt]

        if (utt_feats.shape[0] != utt_adapt_pdfs.shape[0] or
                utt_feats.shape[0] != utt_test_pdfs.shape[0]):
            continue

        feats[spk].append(utt_feats)
        adapt_pdfs[spk].append(utt_adapt_pdfs)
        test_pdfs[spk].append(utt_test_pdfs)

    return generate_batches(params, feats, adapt_pdfs, test_pdfs, num_frames, steps)


def generate_batches(params, feats, adapt_pdfs, test_pdfs, num_frames, steps):
    adapt_x = []
    adapt_y = []
    test_x = []
    test_y = []

    for spk in feats.keys():
        spk_feats = np.concatenate(feats[spk])
        spk_adapt_pdfs = np.concatenate(adapt_pdfs[spk])
        spk_test_pdfs = np.concatenate(test_pdfs[spk])

        for offset in range(0, spk_feats.shape[0] - 2 * num_frames, num_frames):
            adapt_x.append([spk_feats[offset:offset + num_frames]] * steps)
            adapt_y.append([spk_adapt_pdfs[offset:offset + num_frames]] * steps)
            test_x.append(spk_feats[offset + num_frames:offset + 2 * num_frames])
            test_y.append(spk_test_pdfs[offset + num_frames:offset + 2 * num_frames])

    params = np.array([params] * len(adapt_x))
    adapt_x = np.array(adapt_x)
    adapt_y = np.array(adapt_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    return [params, adapt_x, adapt_y, test_x], test_y


def load_utt_to_pdfs(pdfs):
    utt_to_pdfs = {}
    with kaldi_io.SequentialInt32VectorReader(pdfs) as reader:
        for utt, utt_pdfs in reader:
            utt_to_pdfs[utt] = utt_pdfs.reshape((-1, 1))

    return utt_to_pdfs

def load_utt_to_spk(utt2spk):
    spks = {'unk': 0}
    utt_to_spk = {}
    with open(utt2spk, 'r') as f:
        for line in f:
            (utt, spk) = line.split()

            if spk not in spks:
                spks[spk] = len(spks)

            utt_to_spk[utt] = spks[spk]

    return utt_to_spk

