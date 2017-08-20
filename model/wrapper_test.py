from keras.models import Model, Sequential
from keras.layers import Activation, Dense, Input
import numpy as np
import unittest

from wrapper import ModelWrapper


class TestWrapper(unittest.TestCase):

  def testSomething(self):
    model = self.build_model()
    wrapper = self.build_wrapped_model(model)

    params = np.array([
      [1., 0., 0., 1., -1., -1.],
      [1., 0., 0., 1., 0., 0.],
      [2., 0., 0., 2., 1., 1.],
    ])

    x = np.array([
      [1., 1.],
      [1., 1.],
      [1., 1.],
    ])

    expected_result = np.array([
      [0., 0.],
      [1., 1.],
      [3., 3.]
    ])

    prediction = wrapper.predict([params, x])
    np.testing.assert_allclose(expected_result, prediction)

  def testGetAllWeights(self):
    model = self.build_model()
    model.set_weights((np.eye(2), np.zeros(2)))
    wrapper = ModelWrapper(model)

    expected_weights = np.array([1., 0., 0., 1., 0., 0.])
    np.testing.assert_allclose(expected_weights, wrapper.get_all_weights())

  def build_wrapped_model(self, model):
    wrapper = ModelWrapper(model)

    params = Input(shape=(6,))
    x = Input(shape=(2,))
    y = wrapper([params, x])

    return Model(inputs=[params, x], outputs=[y])

  def build_model(self):
    model = Sequential()
    model.add(Dense(2, input_shape=(2,)))
    model.add(Activation('relu'))
    model.compile(loss='mse', optimizer='SGD')

    return model
