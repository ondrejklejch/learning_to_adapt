from keras import backend as K
from keras.activations import get as get_activation
from keras.engine.topology import Layer
from keras.layers import Activation, Dense, Conv1D
from layers import FeatureTransform, LHUC
import numpy as np
from scipy import sparse
import tensorflow as tf


def create_model_wrapper(model, sparse=False, num_sparse_params=10000):
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
    elif isinstance(layer, Activation):
      layers.append({
        "type": "activation",
        "activation": layer.activation.__name__,
        "trainable": layer.trainable,
        "num_params": count_params(layer),
        "weights_shapes": [w.shape for w in layer.get_weights()],
      })

  if not sparse:
    return ModelWrapper(
      feat_dim,
      num_labels,
      num_params,
      loss,
      layers)
  else:
    return SparseModelWrapper(
      num_sparse_params=num_sparse_params,
      feat_dim=feat_dim,
      num_labels=num_labels,
      num_params=num_params,
      loss=loss,
      layers=layers)

def count_params(layer):
  return sum([w.flatten().shape[0] for w in layer.get_weights()])

def reshape_params(shapes, params):
  reshaped_params = []
  offset = 0

  for shape in shapes:
    size = np.prod(shape)
    new_params = params[0, offset:offset + size]
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

  def call(self, inputs):
    if len(inputs) == 3:
        params, trainable_params, x = inputs
        params = self.merge_params(params, trainable_params)
    elif len(inputs) == 2:
        params, x = inputs
    else:
        raise ValueError("Wrong number of inputs")

    offset = 0
    x = x[0]
    for layer in self.layers:
      weights = params[:, offset:offset + layer["num_params"]]
      offset += layer["num_params"]
      x = self.evaluate_layer(layer, weights, x)

    return K.expand_dims(x, 0)

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
    if layer["type"] == "activation":
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


class SparseModelWrapper(ModelWrapper):

    def __init__(self, num_sparse_params, *args, **kwargs):
        super(SparseModelWrapper, self).__init__(**kwargs)

        self.num_sparse_params = num_sparse_params
        self.sample_indices()

    def build(self, input_shapes):
        super(SparseModelWrapper, self).build(input_shapes)

        self.indices = self.add_weight(
            shape=(self.num_trainable_params,),
            name='indices',
            initializer=self.indices_initializer,
            trainable=False,
            dtype='int64',
        )

    def sample_indices(self):
        ratio = float(self.num_sparse_params) / float(self.num_params)

        indices = []
        param_groups = []
        total_indices = 0
        for (s, e) in super(SparseModelWrapper, self).param_groups(trainable_only=False):
            num_indices = int((e - s) * ratio)
            choice = s + np.random.choice(e - s, num_indices, replace=False)
            indices.append(sorted(choice))
            param_groups.append((total_indices, total_indices + num_indices))
            total_indices += num_indices

        self._indices = np.concatenate(indices)
        self._param_groups = param_groups
        self.num_trainable_params = self._indices.shape[0]

    def indices_initializer(self, shape, *args, **kwargs):
        return self._indices

    def get_trainable_params(self, params):
        if not self.built:
            self.build(None)

        return K.reshape(tf.gather(params[0], self.indices), (1, -1))

    def merge_params(self, params, trainable_params):
        mask1 = tf.SparseTensor(
            indices=tf.stack([self.indices, self.indices], axis=-1),
            values=tf.ones_like(self.indices, dtype='float32'),
            dense_shape=[self.num_params, self.num_params])

        mask2 = tf.SparseTensor(
            indices=tf.stack([self.indices, K.arange(0, self.num_trainable_params, dtype='int64')], axis=-1),
            values=tf.ones_like(self.indices, dtype='float32'),
            dense_shape=[self.num_params, self.num_trainable_params])

        old_params = K.transpose(tf.sparse_tensor_dense_matmul(mask1, tf.reshape(params, (-1, 1))))
        new_params = K.transpose(tf.sparse_tensor_dense_matmul(mask2, tf.reshape(trainable_params, (-1, 1))))

        return params - old_params + new_params

    def param_groups(self):
        return self._param_groups

    def get_config(self):
        config = super(SparseModelWrapper, self).get_config()
        config["num_sparse_params"] = self.num_sparse_params

        return config
