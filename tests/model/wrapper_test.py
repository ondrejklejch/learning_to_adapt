from keras.models import Model, Sequential, load_model
from keras.layers import Activation, Dense, Input
import keras.backend as K
import numpy as np
import unittest

from learning_to_adapt.model.layers import FeatureTransform, LHUC
from learning_to_adapt.model.wrapper import create_model_wrapper, get_model_weights, set_model_weights, SparseModelWrapper


class TestWrapper(unittest.TestCase):

  def testForwardPass(self):
    batch_size = 10
    model = self.build_model()
    wrapper = self.build_wrapped_model(model)

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

    prediction = wrapper.predict([params, params, x])
    np.testing.assert_allclose(expected_result, prediction)

  def testForwardPassWithTrainableWeights(self):
    model = self.build_model()
    params = np.array([[1., 1., 0., 0., 1., 0., 0., 1., -1., -1., 1., 1.]])
    set_model_weights(model, params[0])
    for l in model.layers:
      l.trainable = l.name.startswith("dense")

    batch_size = 10
    wrapper = self.build_wrapped_model(model)

    x = np.array([
      [[1., 2.]] * batch_size,
    ])

    expected_result = np.array([
      [[12., 16.]] * batch_size,
    ])

    trainable_params = np.array([[1., 2., 3., 4., 5., 6.]])
    prediction = wrapper.predict([params, trainable_params, x])
    np.testing.assert_allclose(expected_result, prediction)

  def testGetAllWeights(self):
    model = self.build_model()
    model.set_weights((np.ones(2), np.zeros(2), np.eye(2), np.zeros(2), np.ones(2)))
    wrapper = create_model_wrapper(model)

    expected_weights = np.array([1., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 1.])
    np.testing.assert_allclose(expected_weights, get_model_weights(model))

  def testGetParamGroups(self):
    model = self.build_model()
    wrapper = create_model_wrapper(model)

    expected_groups = [(0, 4), (4, 10), (10, 12)]
    self.assertEqual(expected_groups, list(wrapper.param_groups()))

  def testGetParamGroupsWithTrainableParameters(self):
    model = self.build_model()
    for l in model.layers:
      l.trainable = l.name.startswith("dense")
    wrapper = create_model_wrapper(model)

    expected_groups = [(0, 6)]
    self.assertEqual(expected_groups, list(wrapper.param_groups()))

  def testCanSerializeSparseModelWrapper(self):
    model = self.build_model()
    wrapper = self.build_wrapped_model(model, sparse=True)
    original_weights = wrapper.get_weights()

    model_path = "/tmp/sparse_wrapper.h5"
    wrapper.save(model_path)
    reloaded_wrapper = load_model(model_path, custom_objects={'SparseModelWrapper': SparseModelWrapper})
    reloaded_weights = reloaded_wrapper.get_weights()

    np.testing.assert_allclose(original_weights, reloaded_weights)

  def testSparseModelWrapperPredictsCorrectValues(self):
    model = self.build_model()
    x = -356
    params = np.array([[x, 1., x, 0., x, x, 0., 1., x, -1., x, 1.]])
    set_model_weights(model, params[0])

    batch_size = 10
    wrapper = self.build_wrapped_model(model, sparse=True)
    wrapper.set_weights([np.array([0, 2, 4, 5, 8, 10])])

    x = np.array([
      [[1., 1.]] * batch_size,
    ])

    expected_result = np.array([
      [[0., 0.]] * batch_size,
    ])

    trainable_params = np.array([[1., 0., 1., 0., -1., 1.]])
    prediction = wrapper.predict([params, trainable_params, x])
    np.testing.assert_allclose(expected_result, prediction)

  def build_wrapped_model(self, model, sparse=False):
    if sparse:
      wrapper = create_model_wrapper(model, sparse=True, num_sparse_params=6)
    else:
      wrapper = create_model_wrapper(model)

    params = Input(shape=(wrapper.num_params,))
    trainable_params = Input(shape=(wrapper.num_trainable_params,))
    x = Input(shape=K.int_shape(model.inputs[0]))
    y = wrapper([params, trainable_params, x])

    return Model(inputs=[params, trainable_params, x], outputs=[y])

  def build_model(self):
    model = Sequential()
    model.add(FeatureTransform(input_shape=(2,)))
    model.add(Dense(2))
    model.add(Activation('relu'))
    model.add(LHUC())
    model.compile(loss='mse', optimizer='SGD')

    return model
