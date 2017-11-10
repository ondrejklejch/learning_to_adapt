from keras import backend as K
from keras.activations import get as get_activation
from keras.engine.topology import Layer
from keras.layers import Activation, Dense
from layers import FeatureTransform, LHUC
import numpy as np


def create_model_wrapper(model):
  all_params = get_model_weights(model, trainable_only=False)
  trainable_params = get_model_weights(model, trainable_only=True)

  feat_dim = model.layers[0].input_shape[-1]
  num_labels = model.layers[-1].output_shape[-1]
  num_params = len(all_params)
  num_trainable_params = len(trainable_params)
  loss = model.loss

  layers = []
  last_size = 0
  for layer in model.layers:
    if isinstance(layer, Dense):
      layers.append(("dense", layer.units, layer.use_bias, layer.activation.__name__, layer.trainable))
      last_size = layer.units
    if isinstance(layer, FeatureTransform):
      layers.append(("feature_transform", feat_dim, layer.trainable))
      last_size = feat_dim
    if isinstance(layer, LHUC):
      layers.append(("lhuc", last_size, layer.trainable))
    elif isinstance(layer, Activation):
      layers.append(("activation", layer.activation.__name__, layer.trainable))

  return ModelWrapper(
      feat_dim,
      num_labels,
      num_params,
      num_trainable_params,
      loss,
      layers,
      weights=[all_params])

def get_model_weights(model, trainable_only=False):
  weights = []
  for l in model.layers:
    if not trainable_only or l.trainable:
      for w in l.get_weights():
        weights.extend(w.flatten())

  return np.array(weights)

def set_model_weights(model, weights, trainable_only=False):
  for l in model.layers:
    if trainable_only and not l.trainable:
      continue

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

  def __init__(self, feat_dim, num_labels, num_params, num_trainable_params, loss, layers, **kwargs):
    super(ModelWrapper, self).__init__(**kwargs)

    self.feat_dim = feat_dim
    self.num_labels = num_labels
    self.num_params = num_params
    self.num_trainable_params = num_trainable_params
    self.loss = loss
    self.layers = layers

  def build(self, input_shapes):
    self.params = self.add_weight(
        shape=(self.num_params,),
        name='params',
        initializer='zeros',
        trainable=False
    )

    super(ModelWrapper, self).build(input_shapes)

  def call(self, inputs):
    trainable_params, x = inputs

    self.init()
    for layer in self.layers:
      if layer[0] == "dense":
        input_dim = self.last_size
        weights = self.get_weights(layer[-1], self.params, trainable_params, layer[1])
        weights = K.reshape(weights, (-1, input_dim, layer[1]))
        x = K.batch_dot(x, weights, axes=[2, 1])

        if layer[2]:
          bias = K.expand_dims(self.get_weights(layer[-1], self.params, trainable_params), 1)
          x = x + bias

        x = get_activation(layer[3])(x)
      elif layer[0] == "feature_transform":
        rescale = K.expand_dims(self.get_weights(layer[-1], self.params, trainable_params), 1)
        shift = K.expand_dims(self.get_weights(layer[-1], self.params, trainable_params), 1)
        x = x * rescale + shift
      elif layer[0] == "lhuc":
        r = K.expand_dims(self.get_weights(layer[-1], self.params, trainable_params), 1)
        x = x * r
      if layer[0] == "activation":
        x = get_activation(layer[1])(x)

    return x

  def init(self):
    self.last_weight = 0
    self.last_trainable_weight = 0
    self.last_size = self.feat_dim

  def get_weights(self, trainable, all_params, trainable_params, size=1):
    if trainable:
        weights = trainable_params[:, self.last_trainable_weight:self.last_trainable_weight + self.last_size * size]
        self.last_trainable_weight += self.last_size * size
    else:
        weights = K.expand_dims(self.params[self.last_weight:self.last_weight + self.last_size * size], 0)

    self.last_weight += self.last_size * size
    self.last_size = self.last_size if size == 1 else size

    return weights

  def param_groups(self):
    last_weight = 0
    last_size = self.feat_dim
    for layer in self.layers:
      if not layer[-1]:
        continue

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
      "num_trainable_params": self.num_trainable_params,
      "loss": self.loss,
      "layers": self.layers,
    }

