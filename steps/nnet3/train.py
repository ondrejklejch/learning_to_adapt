import sys
import numpy as np

from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from keras.models import load_model, Sequential
from keras.layers import Activation, Conv1D, BatchNormalization
from keras.optimizers import Adam

from learning_to_adapt.model import LHUC, Renorm
from learning_to_adapt.utils import load_dataset
from learning_to_adapt.optimizers import AdamW

import keras
import tensorflow as tf

config = tf.ConfigProto()
config.intra_op_parallelism_threads=1
config.inter_op_parallelism_threads=1
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))


def load_lda(path):
    rows = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip(" []\n")

            if line:
                rows.append(np.fromstring(line, dtype=np.float32, sep=' '))

    matrix = np.array(rows).T

    return matrix[:-1], matrix[-1]


def create_model(hidden_dim=350, lda_path):
    lda, bias = load_lda(lda_path)
    lda = lda.reshape((5, 40, 200))

    model = Sequential()
    model.add(Conv1D(200, 5, strides=1, padding="valid", dilation_rate=1, use_bias=True, input_shape=(None, 40), name="lda", trainable=False, weights=[lda, bias]))
    model.add(Conv1D(hidden_dim, 1, strides=1, padding="valid", dilation_rate=1, use_bias=True, input_shape=(None, 200), name="tdnn1.affine"))
    model.add(Activation("relu", name="tdnn1.relu"))
    model.add(BatchNormalization(name="tdnn1.batchnorm"))
    model.add(LHUC(name="lhuc.tdnn1.batchnorm", trainable=False))
    model.add(Conv1D(hidden_dim, 2, strides=1, padding="valid", dilation_rate=3, use_bias=True, name="tdnn2.affine"))
    model.add(Activation("relu", name="tdnn2.relu"))
    model.add(BatchNormalization(name="tdnn2.batchnorm"))
    model.add(LHUC(name="lhuc.tdnn2.batchnorm", trainable=False))
    model.add(Conv1D(hidden_dim, 2, strides=1, padding="valid", dilation_rate=6, use_bias=True, name="tdnn3.affine"))
    model.add(Activation("relu", name="tdnn3.relu"))
    model.add(BatchNormalization(name="tdnn3.batchnorm"))
    model.add(LHUC(name="lhuc.tdnn3.batchnorm", trainable=False))
    model.add(Conv1D(hidden_dim, 2, strides=1, padding="valid", dilation_rate=9, use_bias=True, name="tdnn4.affine"))
    model.add(Activation("relu", name="tdnn4.relu"))
    model.add(BatchNormalization(name="tdnn4.batchnorm"))
    model.add(LHUC(name="lhuc.tdnn4.batchnorm", trainable=False))
    model.add(Conv1D(hidden_dim, 2, strides=1, padding="valid", dilation_rate=6, use_bias=True, name="tdnn5.affine"))
    model.add(Activation("relu", name="tdnn5.relu"))
    model.add(BatchNormalization(name="tdnn5.batchnorm"))
    model.add(LHUC(name="lhuc.tdnn5.batchnorm", trainable=False))
    model.add(Conv1D(hidden_dim, 1, strides=1, padding="valid", dilation_rate=1, use_bias=True, name="tdnn6.affine"))
    model.add(Activation("relu", name="tdnn6.relu"))
    model.add(BatchNormalization(name="tdnn6.batchnorm"))
    model.add(LHUC(name="lhuc.tdnn6.batchnorm", trainable=False))
    model.add(Conv1D(4208, 1, strides=1, padding="valid", dilation_rate=1, use_bias=True, name="output.affine"))
    model.add(Activation("linear", name="output.log-softmax"))
    model.add(Activation("softmax", name="output"))

    return model

if __name__ == '__main__':
    train_data = sys.argv[1]
    val_data = sys.argv[2]
    utt2spk = sys.argv[3]
    pdfs = sys.argv[4]
    left_context = int(sys.argv[5])
    right_context = int(sys.argv[6])
    lda_path = sys.argv[7]
    output_path = sys.argv[8]

    batch_size = 256
    train_dataset = load_dataset(train_data, utt2spk, pdfs, chunk_size=8, subsampling_factor=1, left_context=left_context, right_context=right_context)
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    train_dataset = train_dataset.prefetch(1024)
    x, y = train_dataset.make_one_shot_iterator().get_next()

    val_dataset = load_dataset(val_data, utt2spk, pdfs, chunk_size=8, subsampling_factor=1, left_context=left_context, right_context=right_context)
    val_dataset = val_dataset.batch(batch_size, drop_remainder=True)
    val_dataset = val_dataset.take(1024).cache().repeat()
    val_x, val_y = val_dataset.make_one_shot_iterator().get_next()

    model = create_model(850, lda_path)
    model.compile(
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        optimizer=Adam(lr=0.0005, amsgrad=True, clipvalue=1.)
    )

    callbacks=[
        CSVLogger(output_path + ".csv"),
        ModelCheckpoint(filepath=output_path + "model.{epoch:02d}.h5", save_best_only=False),
        ModelCheckpoint(filepath=output_path + "model.best.h5", save_best_only=True),
    ]

    model.fit(x, y,
        steps_per_epoch=2000,
        epochs=400,
        validation_data=(val_x, val_y),
        validation_steps=1024,
        callbacks=callbacks
    )
