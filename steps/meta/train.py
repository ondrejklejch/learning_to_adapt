import sys
import numpy as np

from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.models import load_model
from keras.optimizers import Adam

from learning_to_adapt.model import create_meta_learner, create_model_wrapper, get_model_weights, load_model
from learning_to_adapt.utils import load_dataset_for_maml, load_params_generator, load_utt_to_pdfs

import keras
import tensorflow as tf

config = tf.ConfigProto()
config.intra_op_parallelism_threads=8
config.inter_op_parallelism_threads=8
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))


def reshape_eval_data(adapt_x, adapt_y, test_x, test_y):
    return (
        tf.reshape(test_x, (-1, test_x.shape[-2], test_x.shape[-1])),
        tf.reshape(test_y, (-1, test_y.shape[-2], test_y.shape[-1])),
    )


if __name__ == '__main__':
    model_path = sys.argv[1]
    train_feats = sys.argv[2]
    val_feats = sys.argv[3]
    adapt_pdfs = sys.argv[4]
    test_pdfs = sys.argv[5]
    adaptation_type = sys.argv[6]
    output_path = sys.argv[7]
    subsampling_factor = int(sys.argv[8])
    left_context = int(sys.argv[9])
    right_context = int(sys.argv[10])

    num_epochs = 20
    batch_size = 4

    model = load_model(model_path, adaptation_type)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    wrapper = create_model_wrapper(model)
    meta = create_meta_learner(wrapper, meta_learner_type='lr_per_layer')
    meta.compile(
        loss=model.loss,
        optimizer=Adam(),
        metrics=['accuracy']
    )

    model_params = get_model_weights(model)
    utt_to_adapt_pdfs = load_utt_to_pdfs(adapt_pdfs)
    utt_to_test_pdfs = load_utt_to_pdfs(test_pdfs)

    params_dataset = load_params_generator(model_params)
    params_dataset = params_dataset.batch(batch_size, drop_remainder=True)

    train_dataset = load_dataset_for_maml(
        train_feats, utt_to_adapt_pdfs, utt_to_test_pdfs,
        left_context=left_context,
        right_context=right_context,
        adaptation_steps=3
    )
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    train_dataset = train_dataset.prefetch(64)
    train_dataset = tf.data.Dataset.zip((params_dataset, train_dataset))
    train_dataset = train_dataset.make_one_shot_iterator().get_next()
    params, (adapt_x, adapt_y, test_x, test_y) = train_dataset

    val_dataset = load_dataset_for_maml(
        val_feats, utt_to_adapt_pdfs, utt_to_test_pdfs,
        left_context=left_context,
        right_context=right_context,
        adaptation_steps=3
    )
    val_dataset = val_dataset.batch(batch_size, drop_remainder=True)
    val_dataset = eval_dataset = val_dataset.take(32).cache().repeat()
    val_dataset = tf.data.Dataset.zip((params_dataset, val_dataset))
    val_dataset = val_dataset.make_one_shot_iterator().get_next()
    val_params, (val_adapt_x, val_adapt_y, val_test_x, val_test_y) = val_dataset

    # We need to reshape the data for evaluation with the nnet3 model.
    eval_dataset = eval_dataset.map(reshape_eval_data)
    eval_x, eval_y = eval_dataset.make_one_shot_iterator().get_next()

    print "Starting training"
    print "Frame accuracy of the original model is: %.4f" % model.evaluate(eval_x, eval_y, steps=32)[1]

    callbacks = [
        CSVLogger(output_path + "meta.csv"),
        ModelCheckpoint(filepath=output_path + "meta.{epoch:02d}.h5", save_best_only=False, period=10),
        ModelCheckpoint(filepath=output_path + "meta.best.h5", save_best_only=True),
    ]

    meta.fit([params, adapt_x, adapt_y, test_x], test_y,
        steps_per_epoch=256,
        epochs=num_epochs,
        validation_data=([val_params, val_adapt_x, val_adapt_y, val_test_x], val_test_y),
        validation_steps=32,
        callbacks=callbacks
    )

    print meta.get_weights()
    print "Frame accuracy of the adapted model is: %.4f" % meta.evaluate([val_params, val_adapt_x, val_adapt_y, val_test_x], val_test_y, steps=32)[1]
