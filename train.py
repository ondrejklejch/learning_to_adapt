import sys

from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

from model import create_meta_learner, ModelWrapper
from utils import load_data


if __name__ == '__main__':
    feats = sys.argv[1]
    utt2spk = sys.argv[2]
    adapt_pdfs = sys.argv[3]
    test_pdfs = sys.argv[4]

    model = Sequential()
    model.add(Dense(256, use_bias=True, input_shape=(7*3*43,)))
    model.add(Dense(256, use_bias=True))
    model.add(Dense(256, use_bias=True))
    model.add(Dense(256, use_bias=True))
    model.add(Dense(256, use_bias=True))
    model.add(Dense(256, use_bias=True))
    model.add(Dense(3792, use_bias=True))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

    meta = create_meta_learner(model, units=10)
    meta.compile(loss=model.loss, optimizer=Adam(lr=0.1, clipnorm=1.), metrics=['accuracy'])
    meta.summary()

    params = ModelWrapper(model).get_all_weights()
    x, y = load_data(params, feats, utt2spk, adapt_pdfs, test_pdfs)
    meta.fit(x, y, epochs=10, batch_size=1)
