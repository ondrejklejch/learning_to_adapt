import sys
import numpy as np

from keras.models import load_model
from keras.optimizers import Adam

from learning_to_adapt.model import create_meta_learner, create_model_wrapper, get_model_weights, FeatureTransform, LHUC
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

if __name__ == '__main__':
    model_path = sys.argv[1]
    feats = sys.argv[2]
    utt2spk = sys.argv[3]
    adapt_pdfs = sys.argv[4]
    test_pdfs = sys.argv[5]
    output_path = sys.argv[6]

    custom_objects = {
        'FeatureTransform': FeatureTransform,
        'LHUC': LHUC
    }
    model = load_model(model_path, custom_objects=custom_objects)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

    meta = create_meta_learner(model, units=20)
    meta.compile(
        loss=model.loss,
        optimizer=Adam(),
        metrics=['accuracy']
    )
    meta.summary()

    params = get_model_weights(model)
    num_batches, generator = load_data(params, feats, utt2spk, adapt_pdfs, test_pdfs, epochs=1)
    meta.fit_generator(generator, steps_per_epoch=num_batches, epochs=20)
    meta.save(output_path)

    print "Frame accuracy of the original model is: %.4f" % compute_frame_accuracy(model, generator, num_batches)
