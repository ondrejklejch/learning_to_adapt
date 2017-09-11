import sys

from keras.models import load_model
from keras.optimizers import Adam

from model import create_meta_learner, ModelWrapper
from utils import load_data


if __name__ == '__main__':
    model_path = sys.argv[1]
    feats = sys.argv[2]
    utt2spk = sys.argv[3]
    adapt_pdfs = sys.argv[4]
    test_pdfs = sys.argv[5]

    model = load_model(model_path)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

    meta = create_meta_learner(model, units=10)
    meta.compile(
        loss=model.loss,
        optimizer=Adam(lr=0.001, clipnorm=1.),
        metrics=['accuracy']
    )
    meta.summary()

    params = ModelWrapper(model).get_all_weights()
    x, y = load_data(params, feats, utt2spk, adapt_pdfs, test_pdfs)
    meta.fit(x, y, epochs=100, batch_size=1)
