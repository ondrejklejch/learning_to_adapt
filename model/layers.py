import keras.backend as K
from keras.engine.topology import Layer

class LHUC(Layer):
  """
  Straightforward LHUC just adding a scalar with no activation after a layer.
  """
  def __init__(self, **kwargs):
    super(LHUC, self).__init__(**kwargs)

  def build(self, input_shape):
    self.r = self.add_weight(
        shape=(input_shape[1],),
        initializer="ones",
        name="lhuc_weights",
        trainable=self.trainable,
        regularizer=None, constraint=None)

  def call(self, x):
    return x * self.r

  def compute_output_shape(self, input_shape):
    return input_shape


class FeatureTransform(Layer):

  def __init__(self, units, **kwargs):
    super(FeatureTransform, self).__init__(**kwargs)
    self.units = units

  def build(self, input_shape):
    self.rescale = self.add_weight(
      shape=(self.units,),
      initializer="ones",
      name="rescale",
      trainable=self.trainable)

    self.shift = self.add_weight(
      shape=(self.units,),
      initializer="zeros",
      name="shift",
      trainable=self.trainable)

  def call(self, x):
    return x * self.rescale + self.shift

  def compute_output_shape(self, input_shape):
    return input_shape
