from keras import backend as K
from keras.activations import get as get_activation
from keras.engine.topology import Layer
from keras.layers import Activation, Dense, Conv1D
from layers import FeatureTransform, LHUC, Renorm
import numpy as np
from scipy import sparse
import tensorflow as tf


def create_model_wrapper(model):
  all_params = get_model_weights(model)
  feat_dim = model.layers[0].input_shape[-1]
  num_labels = model.layers[-1].output_shape[-1]
  num_params = len(all_params)
  loss = model.loss

  layers = []
  for layer in model.layers:
    if isinstance(layer, Dense):
      layers.append({
        "type": "dense",
        "units": layer.units,
        "use_bias": layer.use_bias,
        "activation": layer.activation.__name__,
        "trainable": layer.trainable,
        "num_params": count_params(layer),
        "weights_shapes": [w.shape for w in layer.get_weights()],
      })
    elif isinstance(layer, Conv1D):
      layers.append({
        "type": "conv1d",
        "filters": layer.filters,
        "kernel_size": layer.kernel_size,
        "strides": layer.strides,
        "padding": layer.padding,
        "dilation_rate": layer.dilation_rate,
        "activation": layer.activation.__name__,
        "use_bias": layer.use_bias,
        "trainable": layer.trainable,
        "num_params": count_params(layer),
        "weights_shapes": [w.shape for w in layer.get_weights()],
      })
    elif isinstance(layer, FeatureTransform):
      layers.append({
        "type": "feature_transform",
        "feat_dim": feat_dim,
        "trainable": layer.trainable,
        "num_params": count_params(layer),
        "weights_shapes": [w.shape for w in layer.get_weights()],
      })
    elif isinstance(layer, LHUC):
      layers.append({
        "type": "lhuc",
        "trainable": layer.trainable,
        "num_params": count_params(layer),
        "weights_shapes": [w.shape for w in layer.get_weights()],
      })
    elif isinstance(layer, Renorm):
      layers.append({
        "type": "renorm",
        "trainable": layer.trainable,
        "num_params": count_params(layer),
        "weights_shapes": [w.shape for w in layer.get_weights()],
      })
    elif isinstance(layer, Activation):
      layers.append({
        "type": "activation",
        "activation": layer.activation.__name__,
        "trainable": layer.trainable,
        "num_params": count_params(layer),
        "weights_shapes": [w.shape for w in layer.get_weights()],
      })

  return ModelWrapper(
    feat_dim,
    num_labels,
    num_params,
    loss,
    layers)

def count_params(layer):
  return sum([w.flatten().shape[0] for w in layer.get_weights()])

def parameter_coordinates(shapes):
  num_params = np.sum([np.prod(shape) for shape in shapes])
  output_dims = [shape[-1] for shape in shapes]

  if not output_dims:
    return []

  output_dim = output_dims[0]
  input_dim = num_params / output_dim
  if not all([dim == output_dim for dim in output_dims]):
    raise ValueError("Can't handle different output dimensions")

  return np.stack([
    np.stack([np.arange(input_dim)] * output_dim).flatten() / float(input_dim),
    np.stack([np.arange(output_dim)] * input_dim).T.flatten() / float(output_dim)
  ], axis=-1)

def reshape_params(shapes, params):
  reshaped_params = []
  offset = 0

  for shape in shapes:
    size = np.prod(shape)
    new_params = params[offset:offset + size]
    new_params = K.reshape(new_params, tuple(shape))
    reshaped_params.append(new_params)
    offset += size

  return reshaped_params

def get_model_weights(model):
  weights = []
  for l in model.layers:
    for w in l.get_weights():
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
    self.num_trainable_params = self.count_trainable_params(layers)
    self.loss = loss
    self.layers = layers

  def count_trainable_params(self, layers):
    return sum([l["num_params"] * l["trainable"] for l in layers])

  def get_trainable_params(self, params):
    trainable_params = []

    offset = 0
    for layer in self.layers:
      if layer["trainable"]:
        trainable_params.append(params[:, offset:offset + layer["num_params"]])

      offset += layer["num_params"]

    return K.concatenate(trainable_params)

  def get_param_coordinates(self):
    coordinates = []
    for layer in self.layers:
      if not layer["trainable"] or layer["num_params"] == 0:
        continue

      coordinates.append(parameter_coordinates(layer["weights_shapes"]))

    return K.constant(np.concatenate(coordinates))

  def call(self, inputs):
    if len(inputs) == 3:
        params, trainable_params, x = inputs
        params = self.merge_params(params, trainable_params)
    elif len(inputs) == 2:
        params, x = inputs
    else:
        raise ValueError("Wrong number of inputs")

    return K.map_fn(self.evaluate_model, [params, x], dtype=K.floatx())

  def evaluate_model(self, inputs):
    params, x = inputs

    offset = 0
    for layer in self.layers:
      weights = params[offset:offset + layer["num_params"]]
      offset += layer["num_params"]
      x = self.evaluate_layer(layer, weights, x)

    return x

  def evaluate_layer(self, layer, weights, x):
    old_weights = weights
    weights = reshape_params(layer["weights_shapes"], weights)

    if layer["type"] == "dense":
      x = K.dot(x, weights[0])

      if layer["use_bias"]:
        x = x + weights[1]

      x = get_activation(layer["activation"])(x)
    elif layer["type"] == "conv1d":
      x = K.conv1d(
        x,
        weights[0],
        strides=layer["strides"][0],
        padding=layer["padding"],
        data_format="channels_last",
        dilation_rate=layer["dilation_rate"][0]
      )

      if layer["use_bias"]:
        x = x + weights[1]

      x = get_activation(layer["activation"])(x)
    elif layer["type"] == "feature_transform":
      x = x * weights[0] + weights[1]
    elif layer["type"] == "lhuc":
      x = x * weights[0]
    elif layer["type"] == "renorm":
      dim = K.cast(K.shape(x)[-1], K.floatx())
      x = K.l2_normalize(x, axis=-1) * K.sqrt(dim)
    elif layer["type"] == "activation":
      x = get_activation(layer["activation"])(x)

    return x

  def merge_params(self, params, trainable_params):
    offset = 0
    trainable_offset = 0
    new_params = []

    for layer in self.layers:
      if layer["trainable"]:
        new_params.append(trainable_params[:, trainable_offset:trainable_offset + layer["num_params"]])
        trainable_offset += layer["num_params"]
        offset += layer["num_params"]
      else:
        new_params.append(params[:, offset:offset + layer["num_params"]])
        offset += layer["num_params"]

    return K.concatenate(new_params)

  def param_groups(self, trainable_only=True):
    offset = 0
    for layer in self.layers:
      if not layer["trainable"] and trainable_only:
        continue

      if layer["num_params"] == 0:
        continue

      yield (offset, offset + layer["num_params"])
      offset += layer["num_params"]

  def compute_output_shape(self, input_shape):
    return input_shape[-1][:-1] + (self.num_labels,)

  def get_config(self):
    return {
      "feat_dim": self.feat_dim,
      "num_labels": self.num_labels,
      "num_params": self.num_params,
      "loss": self.loss,
      "layers": self.layers,
    }
