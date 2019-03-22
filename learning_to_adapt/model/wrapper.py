from collections import defaultdict
from keras import backend as K
from keras.activations import get as get_activation
from keras.engine.topology import Layer
from keras.layers import Input, Activation, Dense, Conv1D, BatchNormalization
from keras.models import Model
from layers import FeatureTransform, LHUC, Renorm, UttBatchNormalization
import numpy as np
from scipy import sparse
import tensorflow as tf


def create_model_wrapper(model, batch_size=1):
  all_params = get_model_weights(model)
  feat_dim = model.layers[0].input_shape[-1]
  num_labels = model.layers[-1].output_shape[-1]
  num_params = len(all_params)
  loss = model.loss

  layers = []
  for layer in model.layers:
    if isinstance(layer, Dense):
      layers.append({
        "name": layer.name,
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
        "name": layer.name,
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
        "name": layer.name,
        "type": "feature_transform",
        "feat_dim": feat_dim,
        "trainable": layer.trainable,
        "num_params": count_params(layer),
        "weights_shapes": [w.shape for w in layer.get_weights()],
      })
    elif isinstance(layer, LHUC):
      layers.append({
        "name": layer.name,
        "type": "lhuc",
        "trainable": layer.trainable,
        "num_params": count_params(layer),
        "weights_shapes": [w.shape for w in layer.get_weights()],
      })
    elif isinstance(layer, Renorm):
      layers.append({
        "name": layer.name,
        "type": "renorm",
        "trainable": layer.trainable,
        "num_params": count_params(layer),
        "weights_shapes": [w.shape for w in layer.get_weights()],
      })
    elif isinstance(layer, UttBatchNormalization):
      layers.append({
        "name": layer.name,
        "type": "batchnorm",
        "trainable": layer.trainable,
        "num_params": count_params(layer),
        "weights_shapes": [w.shape for w in layer.get_weights()],
        "epsilon": layer.epsilon,
      })
    elif isinstance(layer, BatchNormalization):
      layers.append({
        "name": layer.name,
        "type": "standard-batchnorm",
        "trainable": layer.trainable,
        "num_params": count_params(layer),
        "weights_shapes": [w.shape for w in layer.get_weights()],
        "momentum": layer.momentum,
        "epsilon": layer.epsilon,
      })
    elif isinstance(layer, Activation):
      layers.append({
        "name": layer.name,
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
    layers,
    batch_size,
    weights=get_model_stats(model))

def create_model(wrapper, lda=None):
  if lda:
    x = Input(shape=(None, 40))
    y = Conv1D(
      filters=lda.feat_dim * lda.kernel_size,
      kernel_size=lda.kernel_size,
      trainable=False,
      name='lda'
    )(x)
  else:
    x = y = Input(shape=(None, wrapper.feat_dim))

  for l in wrapper.layers:
    if l["type"] == "dense":
      y = Dense(
        units=l["units"],
        use_bias=l["use_bias"],
        activation=l["activation"],
        trainable=l["trainable"],
        name=l.get("name", None)
      )(y)
    elif l["type"] == "conv1d":
      y = Conv1D(
        filters=l["filters"],
        kernel_size=l["kernel_size"],
        strides=l["strides"],
        padding=l["padding"],
        dilation_rate=l["dilation_rate"],
        activation=l["activation"],
        use_bias=l["use_bias"],
        trainable=l["trainable"],
        name=l.get("name", None)
      )(y)
    elif l["type"] == "feature_transform":
      y = FeatureTransform(trainable=l["trainable"], name=l.get("name", None))(y)
    elif l["type"] == "lhuc":
      y = LHUC(trainable=l["trainable"], name=l.get("name", None))(y)
    elif l["type"] == "renorm":
      y = Renorm(name=l.get("name", None))(y)
    elif l["type"] == "batchnorm":
      y = UttBatchNormalization(epsilon=l["epsilon"], trainable=l["trainable"], name=l.get("name", None))(y)
    elif l["type"] == "standard-batchnorm":
      y = BatchNormalization(epsilon=l["epsilon"], trainable=l["trainable"], name=l.get("name", None))(y)
    elif l["type"] == "activation":
      y = Activation(l["activation"], name=l.get("name", None))(y)
    else:
      raise ValueError("Not implemented: %s" % l["type"])

  return Model(inputs=x, outputs=y)

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

def get_model_stats(model):
  stats = []
  for l in model.layers:
    if isinstance(l, BatchNormalization):
      stats.extend(l.get_weights()[2:])

  return stats

def set_model_weights(model, weights, wrapper=None):
  for l in model.layers:
    layer_weights = []
    for w in l.weights:
      num_weights = np.prod(w.shape)
      layer_weights.append(weights[:num_weights].reshape(w.shape))
      weights = weights[num_weights:]

    if isinstance(l, BatchNormalization) and wrapper is not None:
      layer_weights[2] = K.get_session().run(wrapper.moving_means[l.name])
      layer_weights[3] = K.get_session().run(wrapper.moving_vars[l.name])

    l.set_weights(layer_weights)


class ModelWrapper(Layer):
  """
  This wrapper allows to use DNN's parameters as inputs to the model,
  meaning that if we have a model f(x) with parameters w, we can call
  a wrapper g(w, x) such that g(w, x) == f(x).
  """

  def __init__(self, feat_dim, num_labels, num_params, loss, layers, batch_size=4, **kwargs):
    super(ModelWrapper, self).__init__(**kwargs)

    self.feat_dim = feat_dim
    self.num_labels = num_labels
    self.num_params = num_params
    self.num_trainable_params = self.count_trainable_params(layers)
    self.loss = loss
    self.layers = layers
    self.batch_size = batch_size

  def build(self, input_shape):
    self.moving_means = {}
    self.moving_vars = {}

    for layer in self.layers:
      if layer["type"] == "standard-batchnorm":
        self.moving_means[layer["name"]] = self.add_weight(
          shape=layer["weights_shapes"][2],
          name='moving_mean_%s' % layer["name"],
          initializer='zeros',
          trainable=False,
        )

        self.moving_vars[layer["name"]] = self.add_weight(
          shape=layer["weights_shapes"][3],
          name='moving_vars_%s' % layer["name"],
          initializer='ones',
          trainable=False,
        )

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

  def call(self, inputs, training=None):
    if len(inputs) == 3:
        params, trainable_params, x = inputs
        params = self.merge_params(params, trainable_params)
    elif len(inputs) == 2:
        params, x = inputs
    else:
        raise ValueError("Wrong number of inputs")

    self.mean_stats = defaultdict(list)
    self.variance_stats = defaultdict(list)

    outputs = K.stack([self.evaluate_model([params[i], x[i]], training=training) for i in range(self.batch_size)], 0)

    if training is True:
        for layer in self.layers:
            if layer["type"] != "standard-batchnorm":
                continue

            if len(self.mean_stats[layer["name"]]) == 0:
                continue

            moving_mean = self.moving_means[layer["name"]]
            moving_var = self.moving_vars[layer["name"]]

            mean = K.mean(K.stack(self.mean_stats[layer["name"]], 0), 0)
            variance = K.mean(K.stack(self.variance_stats[layer["name"]], 0), 0)

            self.add_update([
              K.moving_average_update(moving_mean, mean, layer["momentum"]),
              K.moving_average_update(moving_var, variance, layer["momentum"]),
            ], inputs)

    return outputs

  def evaluate_model(self, inputs, training=None):
    params, x = inputs

    offset = 0
    for layer in self.layers:
      weights = params[offset:offset + layer["num_params"]]
      offset += layer["num_params"]
      x= self.evaluate_layer(layer, weights, x, training)

    return x

  def evaluate_layer(self, layer, weights, x, training=None):
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
    elif layer["type"] == "batchnorm":
      x = K.normalize_batch_in_training(x, weights[0], weights[1], [0, 1], epsilon=layer["epsilon"])[0]
    elif layer["type"] == "standard-batchnorm":
      def normalize_training():
        normalized_x, mean, variance = K.normalize_batch_in_training(x, weights[0], weights[1], [0, 1], epsilon=layer["epsilon"])

        self.mean_stats[layer["name"]].append(mean)
        self.variance_stats[layer["name"]].append(variance)

        return normalized_x

      def normalize_inference():
        moving_mean = self.moving_means[layer["name"]]
        moving_var = self.moving_vars[layer["name"]]

        return K.batch_normalization(x, moving_mean, moving_var, weights[1], weights[0], epsilon=layer["epsilon"])

      if training is True:
        x = K.in_train_phase(normalize_training, normalize_inference)
      else:
        x = normalize_inference()
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
      "batch_size": self.batch_size
    }
