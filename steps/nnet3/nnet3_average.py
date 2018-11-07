import glob
import sys
import numpy as np

from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.models import load_model, Sequential
from keras.layers import Activation, Conv1D
from keras.optimizers import Adam

from learning_to_adapt.model import FeatureTransform, LHUC, Renorm, create_model_average, create_model_wrapper, get_model_weights, set_model_weights
from learning_to_adapt.utils import load_dataset, load_utt_to_spk, load_utt_to_pdfs
from learning_to_adapt.optimizers import AdamW

import keras
import tensorflow as tf

config = tf.ConfigProto()
config.intra_op_parallelism_threads=1
config.inter_op_parallelism_threads=1
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))


if __name__ == '__main__':
    feats = sys.argv[1]
    utt2spk = sys.argv[2]
    pdfs = sys.argv[3]
    left_context = int(sys.argv[4])
    right_context = int(sys.argv[5])
    output_path = sys.argv[6]

    params = []
    model = None
    models = list(sorted(glob.glob('%s/model.[0-9]*.h5' % output_path)))
    models = models[9::5]
    print models

    for model_path in models:
      model = keras.models.load_model(model_path, custom_objects={'FeatureTransform': FeatureTransform, 'LHUC': LHUC, 'Renorm': Renorm, 'AdamW': AdamW})
      params.append(get_model_weights(model))

    params = np.array(params)
    coeffs = np.ones((1, params.shape[0])) / float(params.shape[0])
    average_model = create_model_average(create_model_wrapper(model), params, coeffs)
    average_model.summary()

    utt_to_spk = load_utt_to_spk(utt2spk)
    utt_to_pdfs = load_utt_to_pdfs(pdfs)

    train_dataset = load_dataset(train_data, utt_to_spk, utt_to_pdfs, chunk_size=8, subsampling_factor=1, left_context=left_context, right_context=right_context)
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    train_dataset = train_dataset.prefetch(1024)
    x, _, y = train_dataset.make_one_shot_iterator().get_next()

    average_model.compile(
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        optimizer='adam'
    )

    average_model.fit(x, y, steps_per_epoch=200, epochs=5)

    coeffs, params = average_model.get_weights()
    params = np.dot(coeffs, params)[0]

    set_model_weights(model, params)
    model.save(output_path + "model.combined.h5")
