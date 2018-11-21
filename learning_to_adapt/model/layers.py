import keras.backend as K
from keras.engine.topology import Layer

class LHUC(Layer):
  """
  Straightforward LHUC just adding a scalar with no activation after a layer.
  """

  def build(self, input_shape):
    self.r = self.add_weight(
        shape=(input_shape[-1],),
        initializer="ones",
        name="lhuc_weights",
        trainable=self.trainable,
        regularizer=None, constraint=None)

  def call(self, x):
    return x * self.r

  def compute_output_shape(self, input_shape):
    return input_shape


class Renorm(Layer):

  def call(self, x):
    dim = K.cast(K.shape(x)[-1], K.floatx())
    return K.l2_normalize(x, axis=-1) * K.sqrt(dim)

  def compute_output_shape(self, input_shape):
    return input_shape


class FeatureTransform(Layer):

  def build(self, input_shape):
    self.rescale = self.add_weight(
      shape=(input_shape[-1],),
      initializer="ones",
      name="rescale",
      trainable=self.trainable)

    self.shift = self.add_weight(
      shape=(input_shape[-1],),
      initializer="zeros",
      name="shift",
      trainable=self.trainable)

  def call(self, x):
    return x * self.rescale + self.shift

  def compute_output_shape(self, input_shape):
    return input_shape


class Multiply(Layer):

  def call(self, inputs):
    return inputs[0] * inputs[1]

  def compute_output_shape(self, input_shapes):
    return input_shapes[0]


class SDBatchNormalization(Layer):

  def __init__(self, num_speakers=9572, momentum=0.99, epsilon=1e-3, **kwargs):
    super(SDBatchNormalization, self).__init__(**kwargs)

    self.num_speakers = num_speakers
    self.momentum = momentum
    self.epsilon = epsilon
    self.axis = -1

  def build(self, input_shapes):
    dim = input_shapes[0][-1]
    shape = (self.num_speakers, dim)

    self.gamma = self.add_weight(
      shape=shape,
      name='gamma',
      initializer='ones')
    self.beta = self.add_weight(
      shape=shape,
      name='beta',
      initializer='zeros')
    self.moving_mean = self.add_weight(
      shape=(dim,),
      name='moving_mean',
      initializer='zeros',
      trainable=False)
    self.moving_variance = self.add_weight(
      shape=(dim,),
      name='moving_variance',
      initializer='ones',
      trainable=False)

    self.built = True

  def call(self, inputs, training=None):
    inputs, spk_id = inputs
    spk_id = K.cast(K.flatten(spk_id)[0], 'int32')

    def normalize_inference():
      return K.normalize_batch_in_training(inputs, self.gamma[spk_id], self.beta[spk_id], [0, 1], epsilon=self.epsilon)[0]

    normed_training, mean, variance = K.normalize_batch_in_training(
      inputs, self.gamma[spk_id], self.beta[spk_id], [0, 1], epsilon=self.epsilon)

    sample_size = K.shape(inputs)[1]
    sample_size = K.cast(sample_size, dtype=K.dtype(inputs))
    variance *= sample_size / (sample_size - (1.0 + self.epsilon))

    self.add_update([
      K.moving_average_update(self.moving_mean, mean, self.momentum),
      K.moving_average_update(self.moving_variance, variance, self.momentum)
    ], inputs)

    # Pick the normalized form corresponding to the training phase.
    return K.in_train_phase(normed_training, normalize_inference, training=training)

  def compute_output_shape(self, input_shapes):
    return input_shapes[0]

  def get_config(self):
    base_config = super(SDBatchNormalization, self).get_config()
    config = {
      'num_speakers': self.num_speakers,
      'momentum': self.momentum,
      'epsilon': self.epsilon,
    }

    return dict(list(base_config.items()) + list(config.items()))


class UttBatchNormalization(Layer):

  def __init__(self, epsilon=1e-3, **kwargs):
    super(UttBatchNormalization, self).__init__(**kwargs)

    self.epsilon = epsilon
    self.axis = -1

  def build(self, input_shapes):
    dim = input_shapes[-1]
    shape = (dim,)

    self.gamma = self.add_weight(
      shape=shape,
      name='gamma',
      initializer='ones')
    self.beta = self.add_weight(
      shape=shape,
      name='beta',
      initializer='zeros')

    self.built = True

  def call(self, inputs, training=None):
    return K.normalize_batch_in_training(inputs, self.gamma, self.beta, [0, 1], epsilon=self.epsilon)[0]

  def compute_output_shape(self, input_shapes):
    return input_shapes

  def get_config(self):
    base_config = super(UttBatchNormalization, self).get_config()
    config = {
      'epsilon': self.epsilon,
    }

    return dict(list(base_config.items()) + list(config.items()))
