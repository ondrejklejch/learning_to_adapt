import itertools
import json
from signal import signal, SIGPIPE, SIG_DFL
import sys

import numpy as np
import keras
import kaldi_io
import tensorflow as tf


config = tf.ConfigProto()
config.intra_op_parallelism_threads=1
config.inter_op_parallelism_threads=1
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

if __name__ == '__main__':
    model = sys.argv[1]
    left_context = int(sys.argv[2])
    right_context = int(sys.argv[3])

    if not model.endswith('.h5'):
        raise TypeError ('Unsupported model type. Please use h5 format. Update Keras if needed')

    m = keras.models.load_model(model)
    with kaldi_io.SequentialBaseFloatMatrixReader("ark:-") as arkIn, \
            kaldi_io.BaseFloatMatrixWriter("ark,t:-") as arkOut:
        signal(SIGPIPE, SIG_DFL)

        for utt, utt_feats in arkIn:
            feats = np.zeros((utt_feats.shape[0] + left_context + right_context, utt_feats.shape[1]))
            feats[:left_context,:] = utt_feats[0]
            feats[-right_context:,:] = utt_feats[-1]
            feats[left_context:-right_context,:] = utt_feats
            feats = np.expand_dims(feats, 0)

            logProbMat = m.predict(feats)[0]
            logProbMat[logProbMat == -np.inf] = -100
            arkOut.write(utt, logProbMat)
