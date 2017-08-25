from keras.layers import Dense
from keras.models import Sequential
import numpy as np
import unittest

from meta import create_meta_learner


class TestLoop(unittest.TestCase):

  def testMetaLearnerCanOverfit(self):
    np.random.seed(0)

    model = self.create_model()
    meta = create_meta_learner(model)
    meta.compile(loss=model.loss, optimizer='adam')

    generator = self.generator()
    history = meta.fit_generator(generator, steps_per_epoch=100, epochs=5)

    loss = history.history["loss"]
    self.assertTrue(loss[0] > loss[-1])
    self.assertTrue(0.02 > loss[-1])

  def create_model(self):
    model = Sequential()
    model.add(Dense(1, use_bias=True, input_shape=(1,)))
    model.compile(loss='mse', optimizer='adam')
    return model

  def generator(self):
    while True:
      params = np.array([[1.0, 0.0]])
      training_feats = np.array([[[[1.], [0.]]] * 5])
      training_labels = np.array([[[[1.], [0.]]] * 5])
      testing_feats = np.array([[[1.], [0.]]])
      testing_labels = np.array([[[1.], [0.]]])

      yield [params, training_feats, training_labels, testing_feats], testing_labels
