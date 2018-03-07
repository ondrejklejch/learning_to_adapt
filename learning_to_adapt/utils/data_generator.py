import math
import numpy as np
import random
import kaldi_io
import collections


def load_data(params, feats, utt2spk, adapt_pdfs, test_pdfs, num_frames=1000, shift=500, chunk_size=50, subsampling_factor=1, left_context=0, right_context=0, return_sequences=False):
    if subsampling_factor != 1:
        raise ValueError('Data generator works only with subsampling_factor=1')

    if not return_sequences:
        raise ValueError('Data generator can return only sequences')

    utts_per_spk = load_utts_per_spk(feats, utt2spk, adapt_pdfs, test_pdfs, subsampling_factor)
    chunks_per_spk = create_chunks_per_spk(utts_per_spk, chunk_size, subsampling_factor, left_context, right_context)
    batches = prepare_batches(params, chunks_per_spk, num_frames, shift, chunk_size, subsampling_factor, return_sequences)

    return (len(batches), infinite_iterator(batches))

def load_utts_per_spk(feats, utt2spk, adapt_pdfs, test_pdfs, subsampling_factor):
    utt_to_adapt_pdfs = load_utt_to_pdfs(adapt_pdfs)
    utt_to_test_pdfs = load_utt_to_pdfs(test_pdfs)
    utt_to_spk = load_utt_to_spk(utt2spk)
    feats_reader = kaldi_io.SequentialBaseFloatMatrixReader(feats)

    utts_per_spk = collections.defaultdict(list)
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

        utts_per_spk[spk].append((
            utt_feats[:utt_subsampled_length * subsampling_factor],
            utt_adapt_pdfs[:utt_subsampled_length],
            utt_test_pdfs[:utt_subsampled_length]
        ))

    return utts_per_spk

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

def create_chunks_per_spk(utts_per_spk, chunk_size, subsampling_factor, left_context, right_context):
    chunks_per_spk = collections.defaultdict(list)
    for spk, utts in utts_per_spk.iteritems():
        for feats, adapt_pdfs, test_pdfs in utts:
            chunks_per_spk[spk].extend(create_chunks(feats, adapt_pdfs, test_pdfs, chunk_size, left_context, right_context, subsampling_factor))

    return chunks_per_spk

def create_chunks(feats, adapt_pdfs, test_pdfs, chunk_size, left_context, right_context, subsampling_factor):
    start, end = trim_silence(test_pdfs)
    if end - start < 2 * chunk_size:
        return []

    chunks = []
    feats = pad_feats(feats, left_context, right_context)
    for offset in get_offsets(start, end, chunk_size):
        chunk_feats = feats[offset:offset + chunk_size + right_context - left_context]
        chunk_adapt_pdfs = adapt_pdfs[offset:offset + chunk_size]
        chunk_test_pdfs = test_pdfs[offset:offset + chunk_size]

        assert chunk_feats.shape[0] == chunk_size + right_context - left_context

        chunks.append((chunk_feats, chunk_adapt_pdfs, chunk_test_pdfs))

    return chunks

def pad_feats(feats, left_context, right_context):
    if left_context == 0 and right_context == 0:
        return feats

    padded_feats = np.zeros((feats.shape[0] - left_context + right_context, feats.shape[1]))
    padded_feats[:-left_context,:] = feats[0]
    padded_feats[-right_context:,:] = feats[-1]
    padded_feats[-left_context:-right_context,:] = feats

    return padded_feats

def trim_silence(pdfs):
    silence_pdfs = set([0,41,43,60,118])

    pdfs = pdfs.flatten()
    for start in range(pdfs.shape[0]):
        if pdfs[start] not in silence_pdfs:
            break

    for end in reversed(range(pdfs.shape[0])):
        if pdfs[end] not in silence_pdfs:
            break

    return start, end



def get_offsets(start, end, window):
    length = end - start
    num_chunks = (length - window) / window
    shift = float(length - window) / num_chunks
    return [start + int(shift * i) for i in range(num_chunks)] + [length - window]

def prepare_batches(params, chunks_per_spk, num_frames, shift, chunk_size, subsampling_factor, return_sequences):
    batches = []
    chunks_per_batch = num_frames / chunk_size
    chunks_shift = int(math.ceil(float(shift) / chunk_size))
    for spk, chunks in chunks_per_spk.iteritems():
        feats = np.stack([x[0] for x in chunks])
        adapt_pdfs = np.stack([x[1] for x in chunks])
        test_pdfs = np.stack([x[2] for x in chunks])

        for offset in range(0, len(chunks) - 2 * chunks_per_batch, chunks_shift):
            params = params

            if return_sequences:
                adapt_x = np.expand_dims(feats[offset:offset + chunks_per_batch], 0)
                adapt_y = np.expand_dims(adapt_pdfs[offset:offset + chunks_per_batch], 0)
                test_x = feats[offset + chunks_per_batch:offset + 2 * chunks_per_batch]
                test_y = test_pdfs[offset + chunks_per_batch:offset + 2 * chunks_per_batch]
            else:
                # TODO: for nnet1 compatibility
                pass

            batches.append((
                [np.expand_dims(x, axis=0) for x in [params, adapt_x, adapt_y, test_x]],
                np.expand_dims(test_y, axis=0)
            ))

    return batches

def infinite_iterator(batches):
    while True:
        random.shuffle(batches)

        for batch in batches:
            yield batch
