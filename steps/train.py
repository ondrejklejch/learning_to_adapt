import sys
import numpy as np

from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam

from learning_to_adapt.model import create_meta_learner, create_model_wrapper, get_model_weights, FeatureTransform, LHUC, Renorm
from learning_to_adapt.utils import load_data


def compute_frame_accuracy(model, generator, num_batches):
    predictions = []
    labels = []

    for i in range(num_batches):
        x, y = next(generator)

        predictions.append(np.argmax(model.predict(x[-1][0]), axis=-1))
        labels.append(y)

    predictions = np.concatenate(predictions).flatten()
    labels = np.concatenate(labels).flatten()

    return np.mean(predictions == labels)

def compute_adapted_frame_accuracy(meta, generator, num_batches):
    predictions = []
    labels = []

    for i in range(num_batches):
        x, y = next(generator)

        predictions.append(np.argmax(meta.predict(x), axis=-1))
        labels.append(y)

    predictions = np.concatenate(predictions).flatten()
    labels = np.concatenate(labels).flatten()

    return np.mean(predictions == labels)


def load_acoustic_model(path, adaptation_type="ALL"):
    custom_objects = {
        'FeatureTransform': FeatureTransform,
        'LHUC': LHUC,
        'Renorm': Renorm,
    }

    model = load_model(model_path, custom_objects=custom_objects)

    if adaptation_type == "LHUC":
        for l in model.layers:
            l.trainable = l.name.startswith("lhuc")

    return model


if __name__ == '__main__':
    model_path = sys.argv[1]
    feats = sys.argv[2]
    utt2spk = sys.argv[3]
    adapt_pdfs = sys.argv[4]
    test_pdfs = sys.argv[5]
    adaptation_type = sys.argv[6]
    output_path = sys.argv[7]

    if len(sys.argv) >= 9:
        input_type = sys.argv[8]
        return_sequences = sys.argv[8] == "sequences"
    else:
        input_type = "frames"
        return_sequences = False

    if len(sys.argv) >= 12:
        subsampling_factor = int(sys.argv[9])
        left_context = int(sys.argv[10])
        right_context = int(sys.argv[11])
    else:
        subsampling_factor = 1
        left_context = 0
        right_context = 0

    model = load_acoustic_model(model_path, adaptation_type)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    model.summary()

    wrapper = create_model_wrapper(model)
    meta = create_meta_learner(wrapper, units=20, input_type=input_type)
    meta.compile(
        loss=model.loss,
        optimizer=Adam(),
        metrics=['accuracy']
    )
    meta.summary()

    params = get_model_weights(model)
    num_train_batches, train_generator, num_val_batches, val_generator = load_data(
        params, feats, utt2spk, adapt_pdfs, test_pdfs,
        validation_speakers=5,
        subsampling_factor=subsampling_factor,
        left_context=left_context,
        right_context=right_context,
        return_sequences=return_sequences)

    print "Frame accuracy on train is: %.4f" % compute_frame_accuracy(model, train_generator, num_train_batches)
    print "Frame accuracy on val is: %.4f" % compute_frame_accuracy(model, val_generator, num_val_batches)

    meta.fit_generator(
        generator=train_generator,
        steps_per_epoch=num_train_batches,
        validation_data=val_generator,
        validation_steps=num_val_batches,
        callbacks=[ModelCheckpoint(filepath=output_path, save_best_only=True)],
        epochs=20)

    print "Frame accuracy of the adapted model is: %.4f" % compute_adapted_frame_accuracy(meta, val_generator, num_val_batches)
