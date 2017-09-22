from keras.models import Model, Sequential
from keras.layers import Activation, Dense, Input
import numpy as np
import unittest

from layers import FeatureTransform, LHUC
from wrapper import create_model_wrapper, get_model_weights


class TestWrapper(unittest.TestCase):

  def testSomething(self):
    batch_size = 10
    model = self.build_model()
    wrapper = self.build_wrapped_model(model, batch_size)

    params = np.array([
      [1., 1., 0., 0., 1., 0., 0., 1., -1., -1., 1., 1.],
      [1., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 1.],
      [1., 1., 0., 0., 2., 0., 0., 2., 1., 1., 1., 1.],
      [1., 1., 0., 0., 1., 2., 3., 4., 5., 6., 1., 1.],
    ])

    x = np.array([
      [[1., 1.]] * batch_size,
      [[1., 1.]] * batch_size,
      [[1., 1.]] * batch_size,
      [[1., 2.]] * batch_size,
    ])

    expected_result = np.array([
      [[0., 0.]] * batch_size,
      [[1., 1.]] * batch_size,
      [[3., 3.]] * batch_size,
      [[12., 16.]] * batch_size
    ])

    prediction = wrapper.predict([params, x])
    np.testing.assert_allclose(expected_result, prediction)

  def testGetAllWeights(self):
    model = self.build_model()
    model.set_weights((np.ones(2), np.zeros(2), np.eye(2), np.zeros(2), np.ones(2)))
    wrapper = create_model_wrapper(model)

    expected_weights = np.array([1., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 1.])
    np.testing.assert_allclose(expected_weights, get_model_weights(model))

  def build_wrapped_model(self, model, batch_size):
    wrapper = create_model_wrapper(model)

    params = Input(shape=(12,))
    x = Input(shape=(batch_size, 2,))
    y = wrapper([params, x])

    return Model(inputs=[params, x], outputs=[y])

  def build_model(self):
    model = Sequential()
    model.add(FeatureTransform(input_shape=(2,)))
    model.add(Dense(2))
    model.add(Activation('relu'))
    model.add(LHUC())
    model.compile(loss='mse', optimizer='SGD')

    return model
