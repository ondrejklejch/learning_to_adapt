from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Activation, Dense
import numpy as np


class ModelWrapper(Layer):
  """
  This wrapper allows to use DNN's parameters as inputs to the model,
  meaning that if we have a model f(x) with parameters w, we can call
  a wrapper g(w, x) such that g(w, x) == f(x).
  """

  def __init__(self, model, **kwargs):
    super(ModelWrapper, self).__init__(**kwargs)
    self.model = model

  def call(self, inputs):
    params, x = inputs

    last_weight = 0
    last_size = K.int_shape(x)[-1]
    for layer in self.model.layers:
      if isinstance(layer, Dense):
        weights = params[:, last_weight:last_weight + last_size * layer.units]
        weights = K.reshape(weights, (-1, last_size, layer.units))
        x = K.batch_dot(x, weights, axes=[2, 1])
        last_weight += last_size * layer.units
        last_size = layer.units

        if layer.use_bias:
          weights = K.expand_dims(params[:, last_weight:last_weight + last_size], 1)
          x = x + weights
          last_weight += last_size

        x = layer.activation(x)
      if isinstance(layer, Activation):
        x = layer.activation(x)

    return x

  def compute_output_shape(self, input_shape):
    return input_shape[1][:-1] + (self.model.output_shape[-1],)

  def get_all_weights(self):
    weights = []
    for w in self.model.get_weights():
      weights.extend(w.flatten())

    return np.array(weights)
