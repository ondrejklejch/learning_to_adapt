from keras import backend as K
from keras.activations import get as get_activation
from keras.engine.topology import Layer
from keras.layers import Activation, Dense
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
    num_trainable_params = 0
    last_size = self.feat_dim

    for layer in layers:
      if layer[0] == "dense":
        num_trainable_params += layer[-1] * (last_size * layer[1] + layer[1] * layer[2])
        last_size = layer[1]
      elif layer[0] == "feature_transform":
        num_trainable_params += layer[-1] * 2 * last_size
      elif layer[0] == "lhuc":
        num_trainable_params += layer[-1] * last_size

    return num_trainable_params

  def get_trainable_params(self, params):
    trainable_params = []

    self.init()
    for layer in self.layers:
      if layer[0] == "dense":
        input_dim = self.last_size
        trainable_params.append((layer[-1], self.get_params_for_layer(params, size=layer[1])))

        if layer[2]:
          trainable_params.append((layer[-1], self.get_params_for_layer(params)))
      elif layer[0] == "feature_transform":
        trainable_params.append((layer[-1], self.get_params_for_layer(params)))
        trainable_params.append((layer[-1], self.get_params_for_layer(params)))
      elif layer[0] == "lhuc":
        trainable_params.append((layer[-1], self.get_params_for_layer(params)))

    return K.concatenate([params for (trainable, params) in trainable_params if trainable])

  def call(self, inputs):
    if len(inputs) == 3:
        params, trainable_params, x = inputs
        params = self.merge_params(params, trainable_params)
    elif len(inputs) == 2:
        params, x = inputs
    else:
        raise ValueError("Wrong number of inputs")

    self.init()
    for layer in self.layers:
      if layer[0] == "dense":
        input_dim = self.last_size
        weights = self.get_params_for_layer(params, layer[1])
        weights = K.reshape(weights, (-1, input_dim, layer[1]))
        x = K.batch_dot(x, weights, axes=[2, 1])

        if layer[2]:
          bias = K.expand_dims(self.get_params_for_layer(params), 1)
          x = x + bias

        x = get_activation(layer[3])(x)
      elif layer[0] == "feature_transform":
        rescale = K.expand_dims(self.get_params_for_layer(params), 1)
        shift = K.expand_dims(self.get_params_for_layer(params), 1)
        x = x * rescale + shift
      elif layer[0] == "lhuc":
        r = K.expand_dims(self.get_params_for_layer(params), 1)
        x = x * r
      if layer[0] == "activation":
        x = get_activation(layer[1])(x)

    return x

  def merge_params(self, params, trainable_params):
    new_params = []

    self.init()
    for layer in self.layers:
      if layer[0] == "dense":
        input_dim = self.last_size
        new_params.append(self.get_params_for_layer(params, size=layer[1], trainable=layer[-1], trainable_params=trainable_params))

        if layer[2]:
          new_params.append(self.get_params_for_layer(params, trainable=layer[-1], trainable_params=trainable_params))
      elif layer[0] == "feature_transform":
        new_params.append(self.get_params_for_layer(params, trainable=layer[-1], trainable_params=trainable_params))
        new_params.append(self.get_params_for_layer(params, trainable=layer[-1], trainable_params=trainable_params))
      elif layer[0] == "lhuc":
        new_params.append(self.get_params_for_layer(params, trainable=layer[-1], trainable_params=trainable_params))

    return K.concatenate(new_params)

  def init(self):
    self.last_weight = 0
    self.last_trainable_weight = 0
    self.last_size = self.feat_dim

  def get_params_for_layer(self, params, size=1, trainable=False, trainable_params=None):
    if trainable:
        weights = trainable_params[:, self.last_trainable_weight:self.last_trainable_weight + self.last_size * size]
        self.last_trainable_weight += self.last_size * size
    else:
        weights = params[:, self.last_weight:self.last_weight + self.last_size * size]

    self.last_weight += self.last_size * size
    self.last_size = self.last_size if size == 1 else size

    return weights

  def param_groups(self, trainable_only=True):
    last_weight = 0
    last_size = self.feat_dim
    for layer in self.layers:
      if not layer[-1] and trainable_only:
        if layer[0] == "dense":
           last_size = layer[1]
        continue

      if layer[0] == "dense":
        if layer[2]:
            yield (last_weight, last_weight + last_size * layer[1] + layer[1])
            last_weight += last_size * layer[1] + layer[1]
            last_size = layer[1]
        else:
            yield (last_weight, last_weight + last_size * layer[1])
            last_weight += last_size * layer[1]
            last_size = layer[1]
      elif layer[0] == "feature_transform":
        yield (last_weight, last_weight + 2 * last_size)
        last_weight += 2 * last_size
      elif layer[0] == "lhuc":
        yield (last_weight, last_weight + last_size)
        last_weight += last_size

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
