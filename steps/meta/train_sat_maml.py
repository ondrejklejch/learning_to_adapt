import sys
import numpy as np

from keras import backend as K
from keras.callbacks import Callback, ModelCheckpoint, CSVLogger, LearningRateScheduler, TensorBoard
from keras.layers import Input, Activation, Conv1D
from keras.models import load_model, Model
from keras.optimizers import Adam

from learning_to_adapt.model import create_maml, create_model_wrapper, get_model_weights, FeatureTransform, LHUC, Renorm, UttBatchNormalization
from learning_to_adapt.utils import load_dataset_for_maml, load_utt_to_pdfs, load_lda

import keras
import tensorflow as tf

config = tf.ConfigProto()
config.intra_op_parallelism_threads=8
config.inter_op_parallelism_threads=8
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))


def create_model(hidden_dim=350, adaptation_type='ALL'):
    feats = x = Input(shape=(None, 200))
    layers = [(1, 1), (2, 3), (2, 6), (2, 9), (2, 6), (1, 1)]
    for i, (kernel_size, dilation_rate) in enumerate(layers):
        name = "tdnn%d" % (i + 1)
        x = Conv1D(hidden_dim, kernel_size=kernel_size, dilation_rate=dilation_rate, activation="relu", name="%s.affine" % name, trainable=adaptation_type == 'ALL')(x)
        x = UttBatchNormalization(name="lhuc.%s.batchnorm" % name, trainable=adaptation_type == 'LHUC')(x)

    y = Conv1D(4208, kernel_size=1, activation="softmax", name="output.affine", trainable=adaptation_type == 'ALL')(x)

    return Model(inputs=[feats], outputs=[y])


class LossWeightScheduler(Callback):

    def __init__(self, num_epochs):
        self.num_epochs = float(num_epochs)
        self.adapted = K.variable(0.0)
        self.original = K.variable(1.0)

    def on_epoch_end(self, epoch, logs={}):
        epoch += 1
        K.set_value(self.original, min(1.0, max(0.0, 1.5 - 2.0 * epoch / self.num_epochs)))
        K.set_value(self.adapted, min(1.0, max(0.0, -0.5 + 2.0 * epoch / self.num_epochs)))


if __name__ == '__main__':
    train_feats = sys.argv[1]
    val_feats = sys.argv[2]
    adapt_pdfs = sys.argv[3]
    test_pdfs = sys.argv[4]
    adaptation_type = sys.argv[5]
    output_path = sys.argv[6]
    subsampling_factor = int(sys.argv[7])
    left_context = int(sys.argv[8])
    right_context = int(sys.argv[9])

    num_epochs = 400
    batch_size = 2
    use_second_order_derivatives = True

    loss_weight_scheduler = LossWeightScheduler(num_epochs=num_epochs)

    model = create_model(850, adaptation_type)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(),
        metrics=['accuracy']
    )
    #model.summary()

    wrapper = create_model_wrapper(model, batch_size=batch_size)
    meta = create_maml(wrapper, get_model_weights(model), use_second_order_derivatives=use_second_order_derivatives, lda_path='lda.txt')
    meta.save(output_path + "meta.graph.h5")
    meta.compile(
        loss={'adapted': model.loss, 'original': model.loss},
        loss_weights={'adapted': loss_weight_scheduler.adapted, 'original': loss_weight_scheduler.original},
        optimizer=Adam(),
        metrics={'adapted': 'accuracy', 'original': 'accuracy'}
    )
    meta.summary()

    utt_to_adapt_pdfs = load_utt_to_pdfs(adapt_pdfs)
    utt_to_test_pdfs = load_utt_to_pdfs(test_pdfs)

    train_dataset = load_dataset_for_maml(
        train_feats, utt_to_adapt_pdfs, utt_to_test_pdfs,
        left_context=left_context,
        right_context=right_context,
        adaptation_steps=3
    )
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    train_dataset = train_dataset.prefetch(64)
    train_dataset = train_dataset.make_one_shot_iterator().get_next()
    adapt_x, adapt_y, test_x, test_y = train_dataset

    print "Train dataset ready"

    val_dataset = load_dataset_for_maml(
        val_feats, utt_to_adapt_pdfs, utt_to_test_pdfs,
        left_context=left_context,
        right_context=right_context,
        adaptation_steps=3
    )
    val_dataset = val_dataset.batch(batch_size, drop_remainder=True)
    val_dataset = eval_dataset = val_dataset.take(128).cache().repeat()
    val_dataset = val_dataset.make_one_shot_iterator().get_next()
    val_adapt_x, val_adapt_y, val_test_x, val_test_y = val_dataset

    print "Val dataset ready"

    callbacks = [
        CSVLogger(output_path + "meta.csv"),
        ModelCheckpoint(filepath=output_path + "meta.{epoch:02d}.h5", save_weights_only=True, save_best_only=False, period=10),
        ModelCheckpoint(filepath=output_path + "meta.best.h5", save_weights_only=True, save_best_only=True, monitor='val_adapted_loss'),
        TensorBoard(output_path + "log"),
        loss_weight_scheduler,
    ]

    print "Starting training"
    meta.fit([adapt_x, adapt_y, test_x], [test_y, test_y],
        steps_per_epoch=1024,
        epochs=num_epochs,
        validation_data=([val_adapt_x, val_adapt_y, val_test_x], [val_test_y, val_test_y]),
        validation_steps=128,
        callbacks=callbacks
    )

    print meta.get_weights()[1]
    print "Frame accuracy of the adapted model is: %.4f" % meta.evaluate([val_adapt_x, val_adapt_y, val_test_x], [val_test_y, val_test_y], steps=128)[1]
