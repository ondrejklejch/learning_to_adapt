from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from model import create_meta_learner, ModelWrapper

np.random.seed(0)

def generator(wrapper):
  params = np.array([wrapper.get_all_weights()])
  for i in range(1000):
    train_x = np.random.normal(size=(1, 5, 1000, 7*3*43))
    train_y = np.random.choice(8927//2, size=(1, 5, 1000, 1))

    test_x = np.random.normal(size=(1, 100, 7*3*43))
    test_y = np.random.choice(8927//2, size=(1, 100, 1))

    yield [params, train_x, train_y, test_x], test_y

model = Sequential()
model.add(Dense(256, use_bias=True, input_shape=(7*3*43,)))
model.add(Dense(256, use_bias=True))
model.add(Dense(256, use_bias=True))
model.add(Dense(256, use_bias=True))
model.add(Dense(256, use_bias=True))
model.add(Dense(256, use_bias=True))
model.add(Dense(8927//2, use_bias=True))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

meta = create_meta_learner(model, units=10)
meta.compile(loss=model.loss, optimizer=Adam(lr=0.1, clipnorm=1.))
meta.summary()
meta.fit_generator(generator(ModelWrapper(model)), steps_per_epoch=10, epochs=5)
