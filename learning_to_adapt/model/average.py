from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Input
from keras.models import Model


def create_model_average(wrapper, params, coeffs):
  feat_dim = wrapper.feat_dim
  num_models = params.shape[0]
  num_params = params.shape[1]

  feats = Input(shape=(None, feat_dim,))
  predictions = ModelAverage(wrapper, num_models, num_params, weights=(coeffs, params))(feats)

  return Model(inputs=[feats], outputs=[predictions])


class ModelAverage(Layer):

  def __init__(self, wrapper, num_models, num_params, **kwargs):
    super(ModelAverage, self).__init__(**kwargs)
    self.wrapper = wrapper
    self.num_models = num_models
    self.num_params = num_params

  def build(self, input_shapes):
    self.params = self.add_weight(
      shape=(self.num_models, self.num_params),
      name='params',
      initializer='zeros',
      trainable=False,
    )

    self.coeffs = self.add_weight(
      shape=(1, self.num_models),
      name='coeffs', 
      initializer='zeros',
      trainable=True,
    )

  def call(self, inputs):
    params = K.dot(self.coeffs, self.params)
    return self.wrapper([params, K.expand_dims(inputs, 0)])[0]
