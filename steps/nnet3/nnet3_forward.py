import itertools
import json
import math
from signal import signal, SIGPIPE, SIG_DFL
import os
import sys

import numpy as np
import keras
import kaldi_io
import tensorflow as tf

from learning_to_adapt.model import FeatureTransform, LHUC, Renorm
from learning_to_adapt.utils import pad_feats
from learning_to_adapt.optimizers import AdamW

config = tf.ConfigProto()
config.intra_op_parallelism_threads=1
config.inter_op_parallelism_threads=1
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))


if __name__ == '__main__':
    model = sys.argv[1]
    counts = sys.argv[2]
    frame_subsampling_factor = int(sys.argv[3])
    left_context = int(sys.argv[4])
    right_context = int(sys.argv[5])

    if len(sys.argv) > 6:
        apply_exp = bool(sys.argv[6])
    else:
        apply_exp = False

    if not model.endswith('.h5'):
        raise TypeError ('Unsupported model type. Please use h5 format. Update Keras if needed')

    ## Load model
    m = keras.models.load_model(model, custom_objects={
        'FeatureTransform': FeatureTransform,
        'LHUC': LHUC,
        'Renorm': Renorm,
        'AdamW': AdamW})

    if os.path.isfile(counts):
        with open(counts, 'r') as f:
            counts = np.fromstring(f.read().strip(" []"), dtype='float32', sep=' ')
        priors = counts / np.sum(counts)
    else:
        priors = 1

    with kaldi_io.SequentialBaseFloatMatrixReader("ark:-") as arkIn, \
            kaldi_io.BaseFloatMatrixWriter("ark,t:-") as arkOut:
        signal(SIGPIPE, SIG_DFL)

        for utt, feats in arkIn:
            feats = np.expand_dims(pad_feats(feats, left_context, right_context), 0)

            logProbMat = np.log(m.predict(feats)[0] / priors)
            logProbMat[logProbMat == -np.inf] = -100

            if apply_exp:
                arkOut.write(utt, np.exp(logProbMat))
            else:
                arkOut.write(utt, logProbMat)
