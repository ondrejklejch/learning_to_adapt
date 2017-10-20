from keras import backend as K
from keras.activations import get as get_activation
from keras.engine.topology import Layer
from keras.layers import Activation, Dense
from layers import FeatureTransform, LHUC
import numpy as np


def create_model_wrapper(model):
  feat_dim = model.layers[0].input_shape[-1]
  num_labels = model.layers[-1].output_shape[-1]
  num_params = len(get_model_weights(model))
  loss = model.loss

  layers = []
  for layer in model.layers:
    if isinstance(layer, Dense):
      layers.append(("dense", layer.units, layer.use_bias, layer.activation.__name__))
    if isinstance(layer, FeatureTransform):
      layers.append(("feature_transform",))
    if isinstance(layer, LHUC):
      layers.append(("lhuc",))
    elif isinstance(layer, Activation):
      layers.append(("activation", layer.activation.__name__))

  return ModelWrapper(feat_dim, num_labels, num_params, loss, layers)

def get_model_weights(model):
  weights = []
  for w in model.get_weights():
    weights.extend(w.flatten())

  return np.array(weights)

def set_model_weights(model, weights):
  for l in model.layers:
    layer_weights = []
    for w in l.weights:
      num_weights = np.prod(w.shape)
      layer_weights.append(weights[:num_weights].reshape(w.shape))
      weights = weights[num_weights:]

    l.set_weights(layer_weights)

class ModelWrapper(Layer):
  """
  This wrapper allows to use DNN's parameters as inputs to the model,
  meaning that if we have a model f(x) with parameters w, we can call
  a wrapper g(w, x) such that g(w, x) == f(x).
  """

  def __init__(self, feat_dim, num_labels, num_params, loss, layers, **kwargs):
    super(ModelWrapper, self).__init__(**kwargs)

    self.feat_dim = feat_dim
    self.num_labels = num_labels
    self.num_params = num_params
    self.loss = loss
    self.layers = layers

  def call(self, inputs):
    params, x = inputs

    last_weight = 0
    last_size = self.feat_dim
    for layer in self.layers:
      if layer[0] == "dense":
        weights = params[:, last_weight:last_weight + last_size * layer[1]]
        weights = K.reshape(weights, (-1, last_size, layer[1]))
        x = K.batch_dot(x, weights, axes=[2, 1])
        last_weight += last_size * layer[1]
        last_size = layer[1]

        if layer[2]:
          weights = K.expand_dims(params[:, last_weight:last_weight + last_size], 1)
          x = x + weights
          last_weight += last_size

        x = get_activation(layer[3])(x)
      elif layer[0] == "feature_transform":
        rescale = K.expand_dims(params[:, last_weight:last_weight + last_size], 1)
        shift = K.expand_dims(params[:, last_weight + last_size:last_weight + 2 * last_size], 1)
        last_weight += 2 * last_size
        x = x * rescale + shift
      elif layer[0] == "lhuc":
        r = K.expand_dims(params[:, last_weight:last_weight + last_size], 1)
        last_weight += last_size
        x = x * r
      if layer[0] == "activation":
        x = get_activation(layer[1])(x)

    return x

  def param_groups(self):
    last_weight = 0
    last_size = self.feat_dim
    for layer in self.layers:
      if layer[0] == "dense":
        yield (last_weight, last_weight + last_size * layer[1])
        last_weight += last_size * layer[1]
        last_size = layer[1]

        if layer[2]:
            yield (last_weight, last_weight + last_size)
            last_weight += last_size
      elif layer[0] == "feature_transform":
        yield (last_weight, last_weight + last_size)
        yield (last_weight + last_size, last_weight + 2 * last_size)
        last_weight += 2 * last_size
      elif layer[0] == "lhuc":
        yield (last_weight, last_weight + last_size)
        last_weight += last_size

  def compute_output_shape(self, input_shape):
    return input_shape[1][:-1] + (self.num_labels,)

  def get_config(self):
    return {
      "feat_dim": self.feat_dim,
      "num_labels": self.num_labels,
      "num_params": self.num_params,
      "loss": self.loss,
      "layers": self.layers,
    }

