import os
import sys

import numpy as np
import keras
import kaldi_io
import tensorflow as tf

from keras.models import Model
from keras.layers import Input
from learning_to_adapt.model import FeatureTransform, LHUC, Renorm, Multiply, SDBatchNormalization, UttBatchNormalization, MAML, ModelWrapper, create_model, create_adapter, set_model_weights


def load_model(path):
    return keras.models.load_model(path, compile=False, custom_objects={
        'FeatureTransform': FeatureTransform,
        'LHUC': LHUC,
        'Renorm': Renorm,
        'Multiply': Multiply,
        'SDBatchNormalization': SDBatchNormalization,
        'UttBatchNormalization': UttBatchNormalization,
        'MAML': MAML,
        'ModelWrapper': ModelWrapper,
    })


if __name__ == '__main__':
    model_in = sys.argv[1]
    weights_in = sys.argv[2]
    model_out = sys.argv[3]
    meta_out = sys.argv[4]

    if not model_in.endswith('.h5') or not model_out.endswith('.h5') or not meta_out.endswith('.h5'):
        raise TypeError ('Unsupported model type. Please use h5 format. Update Keras if needed')

    feats = np.random.normal(size=(1, 33, 40))
    spks = np.zeros((1, 1))

    m_in = load_model(model_in)
    m_in.load_weights(weights_in)
    weights = m_in.get_weights()

    m_out = create_model(m_in.get_layer('maml_1').wrapper)
    set_model_weights(m_out, weights[0][0])

    m_out.save(model_out)
    m_out.summary()

    adapter = create_adapter(m_in.get_layer('maml_1').wrapper, weights[1].reshape((-1, 1)))
    adapter.save(meta_out)
    adapter.summary()

    print weights[1]
