import sys
import numpy as np

from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler
from keras.models import Model
from keras.layers import Input, Activation, Conv1D, Embedding, BatchNormalization
from keras.optimizers import Adam

from learning_to_adapt.model import LHUC, Renorm, Multiply
from learning_to_adapt.utils import load_dataset, load_lda, load_utt_to_spk, load_utt_to_pdfs

import keras
import tensorflow as tf

config = tf.ConfigProto()
config.intra_op_parallelism_threads=1
config.inter_op_parallelism_threads=1
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))


def create_sat_model(hidden_dim=350, lda_path=None, num_spks=None):
    lda, bias = load_lda(lda_path)
    lda = lda.reshape((5, 40, 200))

    feats = Input(shape=(None, 40))
    spk_id = Input(shape=(1,))

    x = Conv1D(200, kernel_size=5, name="lda", trainable=False, weights=[lda, bias])(feats)
    layers = [(1, 1), (2, 3), (2, 6), (2, 9), (2, 6), (1, 1)]
    for i, (kernel_size, dilation_rate) in enumerate(layers):
        name = "tdnn%d" % (i + 1)
        x = Conv1D(hidden_dim, kernel_size=kernel_size, dilation_rate=dilation_rate, activation="relu", name="%s.affine" % name)(x)
        x = BatchNormalization(name="%s.batchnorm" % name)(x)

        lhuc = Embedding(num_spks, hidden_dim, embeddings_initializer='ones', name="lhuc%d" % (i + 1))(spk_id)
        x = Multiply()([x, lhuc])

    y = Conv1D(4208, kernel_size=1, activation="softmax", name="output.affine")(x)

    return Model(inputs=[feats, spk_id], outputs=[y])


if __name__ == '__main__':
    train_data = sys.argv[1]
    val_data = sys.argv[2]
    utt2spk = sys.argv[3]
    pdfs = sys.argv[4]
    left_context = int(sys.argv[5])
    right_context = int(sys.argv[6])
    lda_path = sys.argv[7]
    output_path = sys.argv[8]

    num_epochs = 400
    batch_size = 256
    learning_rate = 0.0015

    utt_to_spk = load_utt_to_spk(utt2spk)
    utt_to_pdfs = load_utt_to_pdfs(pdfs)
    num_spks = max(utt_to_spk.values()) + 1

    train_dataset = load_dataset(train_data, utt_to_spk, utt_to_pdfs, chunk_size=8, subsampling_factor=1, left_context=left_context, right_context=right_context)
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    train_dataset = train_dataset.prefetch(1024)
    x, spk, y = train_dataset.make_one_shot_iterator().get_next()

    val_dataset = load_dataset(val_data, utt_to_spk, utt_to_pdfs, chunk_size=8, subsampling_factor=1, left_context=left_context, right_context=right_context)
    val_dataset = val_dataset.batch(batch_size, drop_remainder=True)
    val_dataset = val_dataset.take(512).cache().repeat()
    val_x, val_spk, val_y = val_dataset.make_one_shot_iterator().get_next()

    model = create_sat_model(850, lda_path, num_spks)
    model.compile(
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        optimizer=Adam(lr=learning_rate, amsgrad=True, clipvalue=1.)
    )

    callbacks=[
        CSVLogger(output_path + "model.csv"),
        ModelCheckpoint(filepath=output_path + "model.{epoch:02d}.h5", save_best_only=False, period=10),
        ModelCheckpoint(filepath=output_path + "model.best.h5", save_best_only=True),
        LearningRateScheduler(lambda epoch, lr: learning_rate - epoch * (learning_rate - learning_rate / 10) / num_epochs, verbose=0)
    ]

    model.fit([x, spk], y,
        steps_per_epoch=2000,
        epochs=num_epochs,
        validation_data=([val_x, val_spk], val_y),
        validation_steps=512,
        callbacks=callbacks
    )
