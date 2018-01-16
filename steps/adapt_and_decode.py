#!/usr/bin/env python

##  Copyright (C) 2016 D S Pavan Kumar
##  dspavankumar [at] gmail [dot] com
##
##  This program is free software: you can redistribute it and/or modify
##  it under the terms of the GNU General Public License as published by
##  the Free Software Foundation, either version 3 of the License, or
##  (at your option) any later version.
##
##  This program is distributed in the hope that it will be useful,
##  but WITHOUT ANY WARRANTY; without even the implied warranty of
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##  GNU General Public License for more details.
##
##  You should have received a copy of the GNU General Public License
##  along with this program.  If not, see <http://www.gnu.org/licenses/>.

import itertools
import json
from signal import signal, SIGPIPE, SIG_DFL
import sys

import numpy as np
import keras
import kaldi_io
import tensorflow as tf

from learning_to_adapt.model import FeatureTransform, LHUC, load_meta_learner, get_model_weights, set_model_weights
from learning_to_adapt.utils import load_utt_to_pdfs

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
    model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=config["lr"]))
    model.fit(x, y, batch_size=256, epochs=config["epochs"], verbose=0)


def adapt_lhuc(model, config, x, y):
    for l in model.layers:
        l.trainable = l.name.startswith('lhuc')

    model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=config["lr"]))
    model.fit(x, y, batch_size=256, epochs=config["epochs"], verbose=0)


def adapt_meta(model, config, x, y):
    params = get_model_weights(model).reshape((1, -1))
    x = x.reshape((1, 1, -1, x.shape[-1]))
    y = y.reshape((1, 1, -1, y.shape[-1]))

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
    priors = sys.argv[6]

    if not model.endswith('.h5'):
        raise TypeError ('Unsupported model type. Please use h5 format. Update Keras if needed')

    ## Load model
    m = keras.models.load_model(model, custom_objects={'FeatureTransform': FeatureTransform, 'LHUC': LHUC})
    p = np.genfromtxt(priors, delimiter=',')

    with kaldi_io.SequentialBaseFloatMatrixReader("ark:-") as arkIn, \
            kaldi_io.BaseFloatMatrixWriter("ark,t:-") as arkOut:
        signal(SIGPIPE, SIG_DFL)

        # Reads adaptation_frames from first utterances to obtain adaptation data.
        utt_to_pdfs = load_utt_to_pdfs(adaptation_pdfs)
        feats = []
        pdfs = []
        utt_buffer = []
        for utt, utt_feats in arkIn:
            feats.append(utt_feats)
            pdfs.append(utt_to_pdfs[utt])
            utt_buffer.append((utt, utt_feats))

            if np.concatenate(feats).shape[0] >= adaptation_frames:
                break

        # Adapts model using adaptation data
        feats = np.concatenate(feats)[:adaptation_frames]
        pdfs = np.concatenate(pdfs)[:adaptation_frames]
        adapt(m, adaptation_method, adaptation_config, feats, pdfs)

        # Decodes everything with the adapted model.
        arkIn = itertools.chain(utt_buffer, arkIn)
        for utt, utt_feats in arkIn:
            logProbMat = np.log(m.predict(utt_feats) / p)
            logProbMat[logProbMat == -np.inf] = -100
            arkOut.write(utt, logProbMat)
