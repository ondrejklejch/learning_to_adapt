import math
import numpy as np
import random
import kaldi_io
import collections


def load_data(params, feats, utt2spk, adapt_pdfs, test_pdfs, num_frames=1000, shift=500, subsampling_factor=1, left_context=0, right_context=0, return_sequences=False):
    utt_to_adapt_pdfs = load_utt_to_pdfs(adapt_pdfs)
    utt_to_test_pdfs = load_utt_to_pdfs(test_pdfs)
    utt_to_spk = load_utt_to_spk(utt2spk)
    feats_reader = kaldi_io.SequentialBaseFloatMatrixReader(feats)

    feats_per_spk = collections.defaultdict(list)
    adapt_pdfs_per_spk = collections.defaultdict(list)
    test_pdfs_per_spk = collections.defaultdict(list)
    for (utt, utt_feats) in feats_reader:
        if utt not in utt_to_adapt_pdfs or utt not in utt_to_test_pdfs:
            continue

        spk = utt_to_spk[utt]
        utt_adapt_pdfs = utt_to_adapt_pdfs[utt]
        utt_test_pdfs = utt_to_test_pdfs[utt]

        utt_subsampled_length = utt_feats.shape[0] / subsampling_factor
        if abs(utt_subsampled_length - utt_adapt_pdfs.shape[0]) > 1:
            continue

        if abs(utt_subsampled_length - utt_test_pdfs.shape[0]) > 1:
            continue

        feats_per_spk[spk].append(utt_feats[:utt_subsampled_length * subsampling_factor])
        adapt_pdfs_per_spk[spk].append(utt_adapt_pdfs[:utt_subsampled_length])
        test_pdfs_per_spk[spk].append(utt_test_pdfs[:utt_subsampled_length])

    chunks_per_spk = collections.defaultdict(list)
    for spk in feats_per_spk.keys():
        feats = np.concatenate(feats_per_spk[spk])
        adapt_pdfs = np.concatenate(adapt_pdfs_per_spk[spk])
        test_pdfs = np.concatenate(test_pdfs_per_spk[spk])

        feats = pad_feats(feats, left_context, right_context)
        chunks_per_spk[spk].extend(create_chunks(feats, adapt_pdfs, test_pdfs, num_frames, shift, left_context, right_context, subsampling_factor))

    batches = []
    chunks_shift = int(math.ceil(num_frames / shift))
    for spk, chunks in chunks_per_spk.iteritems():
        for offset in range(0, len(chunks) - chunks_shift):
            params = params

            if return_sequences:
                adapt_x = np.reshape(chunks[offset][0], (1, 1) + chunks[offset][0].shape)
                adapt_y = np.reshape(chunks[offset][1], (1, 1) + chunks[offset][1].shape)
                test_x = np.expand_dims(chunks[offset + chunks_shift][0], 0)
                test_y = np.expand_dims(chunks[offset + chunks_shift][2], 0)
            else:
                adapt_x = np.expand_dims(chunks[offset][0], 0)
                adapt_y = np.expand_dims(chunks[offset][1], 0)
                test_x = chunks[offset + chunks_shift][0]
                test_y = chunks[offset + chunks_shift][2]

            batches.append((
                [np.expand_dims(x, axis=0) for x in [params, adapt_x, adapt_y, test_x]],
                np.expand_dims(test_y, axis=0)
            ))

    return (len(batches), infinite_iterator(batches))

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

def pad_feats(feats, left_context, right_context):
    if left_context == 0 and right_context == 0:
        return feats

    padded_feats = np.zeros((feats.shape[0] - left_context + right_context, feats.shape[1]))
    padded_feats[:-left_context,:] = feats[0]
    padded_feats[-right_context:,:] = feats[-1]
    padded_feats[-left_context:-right_context,:] = feats

    return padded_feats


def create_chunks(feats, adapt_pdfs, test_pdfs, num_frames, shift, left_context, right_context, subsampling_factor):
    chunks = []
    for offset in range(-left_context, feats.shape[0] - num_frames - right_context, shift):
        pdfs_start = int(math.floor((offset + left_context) / float(subsampling_factor)))
        pdfs_end = int(pdfs_start + math.ceil(num_frames / float(subsampling_factor)))

        chunk_feats = feats[offset + left_context:offset + num_frames + right_context]
        chunk_adapt_pdfs = adapt_pdfs[pdfs_start:pdfs_end]
        chunk_test_pdfs = test_pdfs[pdfs_start:pdfs_end]

        chunks.append((chunk_feats, chunk_adapt_pdfs, chunk_test_pdfs))

    return chunks

def infinite_iterator(batches):
    while True:
        random.shuffle(batches)

        for batch in batches:
            yield batch
