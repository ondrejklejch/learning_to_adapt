import sys
import numpy as np

from keras.models import load_model
from keras.optimizers import Adam

from learning_to_adapt.model import create_meta_learner, create_model_wrapper, get_model_weights, FeatureTransform, LHUC
from learning_to_adapt.utils import load_data


def compute_frame_accuracy(model, x, y):
    predictions = []

    for i in range(x.shape[0]):
        predictions.append(np.argmax(model.predict(x[i]), axis=-1))

    predictions = np.concatenate(predictions).flatten()
    y = np.concatenate(y).flatten()

    return np.mean(predictions == y)

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
    x, y = load_data(params, feats, utt2spk, adapt_pdfs, test_pdfs, steps=1)
    meta.fit(x, y, epochs=5, batch_size=1, shuffle=True)
    meta.save(output_path)

    print "Frame accuracy of the original model is: %.4f" % compute_frame_accuracy(model, x[-1], y)
