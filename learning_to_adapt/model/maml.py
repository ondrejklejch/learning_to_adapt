import numpy as np

from keras import backend as K
from keras import losses
from keras.engine import InputSpec
from keras.engine.topology import Layer
from keras.initializers import Ones, Zeros, Constant
from keras.layers import Activation, Input, GaussianNoise, deserialize
from keras.models import Model, load_model
from keras.regularizers import l2

from loop import rnn
from meta import LearningRatePerLayerMetaLearner

def create_maml(wrapper, weights, num_steps=3, use_second_order_derivatives=False, learning_rate=0.001):
  weights = weights.reshape((1, -1))
  learning_rates = np.array([learning_rate] * len(list(wrapper.param_groups())))

  feat_dim = wrapper.feat_dim
  training_feats = Input(shape=(num_steps, None, None, feat_dim,))
  training_labels = Input(shape=(num_steps, None, None, 1,), dtype='int32')
  testing_feats = Input(shape=(None, None, feat_dim,))

  maml = MAML(wrapper, num_steps, use_second_order_derivatives, weights=[weights, learning_rates])
  original_params, adapted_params = tuple(maml([training_feats, training_labels]))

  original_predictions = Activation('linear', name='original')(wrapper([original_params, testing_feats]))
  adapted_predictions = Activation('linear', name='adapted')(wrapper([adapted_params, testing_feats]))

  return Model(
    inputs=[training_feats, training_labels, testing_feats],
    outputs=[adapted_predictions, original_predictions]
  )

def create_adapter(wrapper, learning_rates):
  num_params = wrapper.num_params
  feat_dim = wrapper.feat_dim

  params = Input(shape=(num_params,))
  training_feats = Input(shape=(None, None, None, feat_dim,))
  training_labels = Input(shape=(None, None, None, 1,))

  meta = LearningRatePerLayerMetaLearner(wrapper, weights=[learning_rates])
  adapted_params = meta([training_feats, training_labels, params])

  return Model(inputs=[params, training_feats, training_labels], outputs=[adapted_params])

class MAML(Layer):

  def __init__(self, wrapper, num_steps=3, use_second_order_derivatives=False, **kwargs):
    super(MAML, self).__init__(**kwargs)

    self.wrapper = wrapper
    self.num_steps = num_steps
    self.use_second_order_derivatives = use_second_order_derivatives

    self.param_groups = list(wrapper.param_groups())
    self.num_param_groups = len(self.param_groups)
    self.num_params = self.wrapper.num_params
    self.num_trainable_params = self.wrapper.num_trainable_params

  def build(self, input_shapes):
    self.params = self.add_weight(
      shape=(1, self.num_params),
      name='params',
      initializer='uniform',
    )

    self.learning_rate = self.add_weight(
      shape=(self.num_param_groups,),
      name='learning_rate',
      initializer='zeros',
    )

  def call(self, inputs):
    feats, labels = inputs

    self.repeated_params = K.squeeze(K.repeat(self.params, K.shape(feats)[0]), 0)
    trainable_params = self.wrapper.get_trainable_params(self.repeated_params)

    for i in range(self.num_steps):
      trainable_params = self.step(feats[:,i], labels[:,i], trainable_params)

    return [self.repeated_params, self.wrapper.merge_params(self.repeated_params, trainable_params)]

  def step(self, feats, labels, params):
    gradients = self.compute_gradients(params, feats, labels)

    new_params = []
    for param_group, indices in enumerate(self.param_groups):
      s, e = indices
      new_params.append(params[:, s:e] - self.learning_rate[param_group] * gradients[:, s:e])

    return K.concatenate(new_params, axis=1)

  def compute_gradients(self, trainable_params, feats, labels):
    predictions = self.wrapper([self.repeated_params, trainable_params, feats])

    if self.use_second_order_derivatives:
      loss = K.mean(losses.get('categorical_crossentropy')(K.squeeze(K.one_hot(labels, 4208), 3), predictions), axis=[1,2])
      return K.squeeze(K.gradients(loss, [trainable_params]), 0)
    else:
      loss = K.mean(losses.get(self.wrapper.loss)(labels, predictions), axis=[1,2])
      return K.stop_gradient(K.squeeze(K.gradients(loss, [trainable_params]), 0))

  def compute_output_shape(self, input_shape):
    return [(input_shape[0][0], self.num_params), (input_shape[0][0], self.num_params)]

  def get_config(self):
    return {
      'wrapper': {
        'class_name': self.wrapper.__class__.__name__,
        'config': self.wrapper.get_config()
      },
      'num_steps': self.num_steps,
      'use_second_order_derivatives': self.use_second_order_derivatives,
    }

  @classmethod
  def from_config(cls, config, custom_objects=None):
    wrapper = deserialize(config.pop('wrapper'), custom_objects=custom_objects)

    return cls(wrapper, config.get('num_steps', 3), config.get('use_second_order_derivatives', False))
