import numpy as np
import kaldi_io
import collections


def load_data(params, feats, utt2spk, adapt_pdfs, test_pdfs, num_frames=1000, shift=500, batch_size=1000, epochs=1):
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

    return (
        count_batches(feats, num_frames, shift),
        generate_batches(params, feats, adapt_pdfs, test_pdfs, num_frames, shift, batch_size, epochs)
    )

def count_batches(feats, num_frames, shift):
    num_batches = 0

    for spk in feats.keys():
        spk_feats = np.concatenate(feats[spk])
        num_batches += int((spk_feats.shape[0] - 2 * num_frames) / shift) + 1

    return num_batches

def generate_batches(params, feats, adapt_pdfs, test_pdfs, num_frames, shift, batch_size, epochs):
    all_feats = []
    all_adapt_pdfs = []
    all_test_pdfs = []
    last_size = 0
    offsets = []

    for spk in feats.keys():
        spk_feats = np.concatenate(feats[spk])
        spk_adapt_pdfs = np.concatenate(adapt_pdfs[spk])
        spk_test_pdfs = np.concatenate(test_pdfs[spk])

        max_offset = int((spk_feats.shape[0] - 2 * num_frames) / shift)
        size = max_offset * shift + 2 * num_frames

        all_feats.append(spk_feats[:size])
        all_adapt_pdfs.append(spk_adapt_pdfs[:size])
        all_test_pdfs.append(spk_test_pdfs[:size])

        for offset in range(max_offset + 1):
            offsets.append(last_size + offset * shift)

        last_size += size

    feats = np.concatenate(all_feats)
    adapt_pdfs = np.concatenate(all_adapt_pdfs)
    test_pdfs = np.concatenate(all_test_pdfs)
    max_offset = int((feats.shape[0] - 2 * num_frames) / shift) + 1

    while True:
        for offset in np.random.permutation(offsets):
            x = feats[offset:offset + num_frames]
            y = adapt_pdfs[offset:offset + num_frames]
            current_adapt_x = []
            current_adapt_y = []

            for _ in range(epochs):
                for i in range(0, num_frames, batch_size):
                    current_adapt_x.append(x[i:i + batch_size])
                    current_adapt_y.append(y[i:i + batch_size])

            params = params
            adapt_x = np.array(current_adapt_x)
            adapt_y = np.array(current_adapt_y)
            test_x = feats[offset + num_frames:offset + 2 * num_frames]
            test_y = test_pdfs[offset + num_frames:offset + 2 * num_frames]

            yield (
                [np.expand_dims(x, axis=0) for x in [params, adapt_x, adapt_y, test_x]],
                np.expand_dims(test_y, axis=0)
            )

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

