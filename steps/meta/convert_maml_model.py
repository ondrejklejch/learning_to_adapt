import os
import sys

import numpy as np
import keras
import kaldi_io
import tensorflow as tf

from keras.models import Model
from keras.layers import Input
from learning_to_adapt.model import FeatureTransform, LDA, LHUC, Renorm, Multiply, SDBatchNormalization, UttBatchNormalization, MAML, ModelWrapper, create_maml, create_model, create_adapter, create_model_wrapper, set_model_weights


def load_model(path):
    return keras.models.load_model(path, compile=False, custom_objects={
        'FeatureTransform': FeatureTransform,
        'LDA': LDA,
        'LHUC': LHUC,
        'Renorm': Renorm,
        'Multiply': Multiply,
        'SDBatchNormalization': SDBatchNormalization,
        'UttBatchNormalization': UttBatchNormalization,
        'MAML': MAML,
        'ModelWrapper': ModelWrapper,
    })


def converted_models_produce_correct_output(m_in, m_out):
    # Test that converted models
    adapt_x = np.random.normal(size=(1, 3, 20, 78, 40))
    adapt_y = np.ones((1, 3, 20, 50, 1))
    test_x = np.random.normal(size=(1, 20, 78, 40))

    # Workaround for MAML models with wrong input dimensions
    maml = m_in.get_layer('maml_1')
    maml.wrapper.batch_size = 1
    m_in = create_maml(maml.wrapper, weights[2], maml.num_steps, maml.use_second_order_derivatives)
    m_in.load_weights(weights_in)

    reference_predictions = m_in.predict([adapt_x, adapt_y, test_x])[1]
    test_predictions = m_out.predict(test_x[0])

    return np.allclose(reference_predictions[0][0,:5,:5], test_predictions[0,:5,:5])

if __name__ == '__main__':
    model_in = sys.argv[1]
    weights_in = sys.argv[2]
    model_out = sys.argv[3]
    meta_out = sys.argv[4]

    if not model_in.endswith('.h5') or not model_out.endswith('.h5') or not meta_out.endswith('.h5'):
        raise TypeError ('Unsupported model type. Please use h5 format. Update Keras if needed')

    m_in = load_model(model_in)
    m_in.load_weights(weights_in)
    weights = m_in.get_weights()

    try:
      lda = m_in.get_layer('lda_1')
      model_weights = np.concatenate([weights[0].flatten(), weights[1].flatten(), weights[2].flatten()])
      maml_weights = weights[3].reshape((-1, 1))
    except ValueError:
      lda = None
      model_weights = weights[0][0]
      maml_weights = weights[1].reshape((-1, 1))

    m_out = create_model(m_in.get_layer('maml_1').wrapper, m_in.get_layer('lda_1'))
    set_model_weights(m_out, model_weights)

    assert converted_models_produce_correct_output(m_in, m_out)

    m_out.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    m_out.save(model_out)
    m_out.summary()

    adapter = create_adapter(create_model_wrapper(m_out), maml_weights)
    adapter.save(meta_out)
    adapter.summary()

    print maml_weights
