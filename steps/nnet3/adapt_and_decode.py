import itertools
import json
import math
from signal import signal, SIGPIPE, SIG_DFL
import sys

import numpy as np
import keras
import kaldi_io
import tensorflow as tf

from learning_to_adapt.model import FeatureTransform, LHUC, Renorm, UttBatchNormalization, load_meta_learner, get_model_weights, set_model_weights
from learning_to_adapt.utils import load_utt_to_pdfs, pad_feats, create_chunks

config = tf.ConfigProto()
config.intra_op_parallelism_threads=1
config.inter_op_parallelism_threads=1
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

def adapt(model, method, config, x, y):
    config = load_config(config)

    if method == "ALL":
        adapt_all(model, config, x, y)

    if method == "LHUC":
        adapt_lhuc(model, config, x, y)

    if method == "META":
        adapt_meta(model, config, x, y)


def adapt_all(model, config, x, y):
    adaptation_frames = float(sys.argv[4])
    lr = 2.5e-5 * adaptation_frames / 1000.

    model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=lr))
    model.fit(x, y, batch_size=1024, epochs=3, verbose=0)


def adapt_lhuc(model, config, x, y):
    model.trainable = True
    for l in model.layers:
        l.trainable = l.name.startswith('lhuc')

        if l.name.startswith('lhuc'):
            l.trainable_weights = l.weights

    adaptation_frames = float(sys.argv[4])
    lr = 0.7 * adaptation_frames / 1000.

    model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=lr))
    model.fit(x, y, batch_size=1024, epochs=3, verbose=0)


def adapt_meta(model, config, x, y):
    params = get_model_weights(model).reshape((1, -1))
    x = x.reshape((1, 1) + x.shape)
    y = y.reshape((1, 1) + y.shape)

    epochs = config.get("epochs", 1)
    x = np.repeat(x, epochs, axis=1)
    y = np.repeat(y, epochs, axis=1)

    meta = load_meta_learner(model, config["model"])
    new_params = meta.predict([params, x, y])
    set_model_weights(model, new_params[0])


def load_config(config):
    with open(config, 'r') as f:
        return json.load(f)

if __name__ == '__main__':
    adaptation_method = sys.argv[1]
    adaptation_config = sys.argv[2]
    adaptation_pdfs = sys.argv[3]
    adaptation_frames = int(sys.argv[4])
    model = sys.argv[5]
    counts = sys.argv[6]
    frame_subsampling_factor = int(sys.argv[7])
    left_context = int(sys.argv[8])
    right_context = int(sys.argv[9])
    chunk_size = 10

    if not model.endswith('.h5'):
        raise TypeError ('Unsupported model type. Please use h5 format. Update Keras if needed')

    ## Load model
    m = keras.models.load_model(model, custom_objects={'FeatureTransform': FeatureTransform, 'LHUC': LHUC, 'Renorm': Renorm, 'UttBatchNormalization': UttBatchNormalization})

    with open(counts, 'r') as f:
        counts = np.fromstring(f.read().strip(" []"), dtype='float32', sep=' ')
    priors = counts / np.sum(counts)

    with kaldi_io.SequentialBaseFloatMatrixReader("ark:-") as arkIn, \
            kaldi_io.BaseFloatMatrixWriter("ark,t:-") as arkOut:
        signal(SIGPIPE, SIG_DFL)

        # Reads adaptation_frames from first utterances to obtain adaptation data.
        num_chunks = adaptation_frames / chunk_size
        utt_to_pdfs = load_utt_to_pdfs(adaptation_pdfs)

        chunks = []
        utt_buffer = []
        for utt, feats in arkIn:
            utt_buffer.append((utt, feats))

            if utt not in utt_to_pdfs:
                continue

            pdfs = utt_to_pdfs[utt]
            chunks.extend(create_chunks(feats, pdfs, pdfs, chunk_size, left_context, right_context, frame_subsampling_factor, trim_silence=False))

            if len(chunks) >= num_chunks:
                break

        # Adapts model using adaptation data
        feats = np.stack([x[0] for x in chunks[:num_chunks]])
        pdfs = np.stack([x[1] for x in chunks[:num_chunks]])

        before = np.mean(np.argmax(m.predict(feats), axis=-1).flatten() == pdfs.flatten())
        print >> sys.stderr, "BEFORE", before
        print >> sys.stderr, "SILENCE", len([x for x in pdfs.flatten() if x in set([0, 118, 41, 43, 60])])
        adapt(m, adaptation_method, adaptation_config, feats, pdfs)

        after = np.mean(np.argmax(m.predict(feats), axis=-1).flatten() == pdfs.flatten())
        print >> sys.stderr, "AFTER", after
        print >> sys.stderr, "IMPROVEMENT", after - before

        # Decodes everything with the adapted model.
        arkIn = itertools.chain(utt_buffer, arkIn)
        for utt, feats in arkIn:
            feats = np.expand_dims(pad_feats(feats, left_context, right_context), 0)

            logProbMat = np.log(m.predict(feats)[0] / priors)
            logProbMat[logProbMat == -np.inf] = -100
            arkOut.write(utt, logProbMat)
