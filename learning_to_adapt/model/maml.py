import numpy as np

from keras import backend as K
from keras import losses
from keras.engine import InputSpec
from keras.engine.topology import Layer
from keras.initializers import Ones, Zeros, Constant
from keras.layers import Activation, Input, GaussianNoise, deserialize
from keras.models import Model, load_model
from keras.regularizers import l2

from learning_to_adapt.utils import load_lda

from loop import rnn
from layers import LDA

def create_maml(wrapper, weights, num_steps=3, use_second_order_derivatives=False, use_lr_per_step=False, use_kld_regularization=False, learning_rate=0.001, lda_path=None):
  if lda_path is not None:
    lda, bias = load_lda(lda_path)
    lda = lda.reshape((5, 40, 200))
  else:
    lda = np.eye(200, 200).reshape((5, 40, 200))
    bias = np.zeros(200)

  weights = weights.reshape((1, -1))
  kld = np.zeros((1,))

  if use_lr_per_step:
    learning_rates = learning_rate * np.ones((num_steps, len(list(wrapper.param_groups()))))
  else:
    learning_rates = learning_rate * np.ones((len(list(wrapper.param_groups())),))

  if use_kld_regularization:
    maml_weights = [weights, learning_rates, kld]
  else:
    maml_weights = [weights, learning_rates]

  feat_dim = 40
  training_feats = Input(shape=(num_steps, 20, 78, feat_dim,))
  training_labels = Input(shape=(num_steps, None, None, 1,), dtype='int32')
  testing_feats = Input(shape=(None, None, feat_dim,))

  lda = LDA(feat_dim=feat_dim, kernel_size=5, weights=[lda, bias], trainable=False)
  maml = MAML(wrapper, num_steps, use_second_order_derivatives, use_lr_per_step, use_kld_regularization, train_params=True, weights=maml_weights)
  original_params, adapted_params = tuple(maml([lda(training_feats), training_labels]))

  original_predictions = Activation('linear', name='original')(wrapper([original_params, lda(testing_feats)], training=True))
  adapted_predictions = Activation('linear', name='adapted')(wrapper([adapted_params, lda(testing_feats)], training=False))

  return Model(
    inputs=[training_feats, training_labels, testing_feats],
    outputs=[adapted_predictions, original_predictions]
  )

def create_adapter(wrapper, num_steps, use_lr_per_step, use_kld_regularization, weights):
  num_params = wrapper.num_params
  feat_dim = wrapper.feat_dim

  params = Input(shape=(num_params,))
  training_feats = Input(shape=(None, None, None, feat_dim,))
  training_labels = Input(shape=(None, None, None, 1,))

  meta = MAML(wrapper, num_steps, use_lr_per_step=use_lr_per_step, use_kld_regularization=use_kld_regularization, train_params=False, weights=weights)
  adapted_params = meta([training_feats, training_labels, params])[1]

  return Model(inputs=[params, training_feats, training_labels], outputs=[adapted_params])

class MAML(Layer):

  def __init__(self, wrapper, num_steps=3, use_second_order_derivatives=False, use_lr_per_step=False, use_kld_regularization=False, train_params=True, correctly_serialized=True, **kwargs):
    super(MAML, self).__init__(**kwargs)

    self.wrapper = wrapper
    self.num_steps = num_steps
    self.use_second_order_derivatives = use_second_order_derivatives
    self.use_lr_per_step = use_lr_per_step
    self.use_kld_regularization = use_kld_regularization
    self.train_params = train_params
    self.correctly_serialized = correctly_serialized

    self.param_groups = list(wrapper.param_groups())
    self.num_param_groups = len(self.param_groups)
    self.num_params = self.wrapper.num_params
    self.num_trainable_params = self.wrapper.num_trainable_params

  def build(self, input_shapes):
    if self.train_params:
      self.params = self.add_weight(
        shape=(1, self.num_params),
        name='params',
        initializer='uniform',
      )

    if self.use_lr_per_step:
      self.learning_rate = self.add_weight(
        shape=(self.num_steps, self.num_param_groups,),
        name='learning_rate',
        initializer='zeros',
      )
    else:
      self.learning_rate = self.add_weight(
        shape=(self.num_param_groups,),
        name='learning_rate',
        initializer='zeros',
      )

    if self.use_kld_regularization:
      self.kld_weight = self.add_weight(
        shape=(1,),
        name='kld',
        initializer='zeros',
        trainable=self.use_kld_regularization
      )

  def call(self, inputs):
    if self.train_params:
      feats, labels = inputs
      params = self.params
    else:
      feats, labels, params = inputs

    self.repeated_params = K.squeeze(K.repeat(params, K.shape(feats)[0]), 0)
    trainable_params = self.wrapper.get_trainable_params(self.repeated_params)

    if self.use_kld_regularization:
      self.original_predictions = self.wrapper([self.repeated_params, feats[:,0]], training=False)

    for i in range(self.num_steps):
      trainable_params = self.step(feats[:,i], labels[:,i], trainable_params, step=i)

    return [self.repeated_params, self.wrapper.merge_params(self.repeated_params, trainable_params)]

  def step(self, feats, labels, params, step):
    gradients = self.compute_gradients(params, feats, labels)

    new_params = []
    for param_group, indices in enumerate(self.param_groups):
      s, e = indices

      if self.use_lr_per_step:
        learning_rate = self.learning_rate[step, param_group]
      else:
        learning_rate = self.learning_rate[param_group]

      new_params.append(params[:, s:e] - learning_rate * gradients[:, s:e])

    return K.concatenate(new_params, axis=1)

  def compute_gradients(self, trainable_params, feats, labels):
    predictions = self.wrapper([self.repeated_params, trainable_params, feats], training=False)
    kld_gradients = self.compute_kld_gradients(trainable_params, predictions)

    if self.use_second_order_derivatives:
      loss = K.sum(losses.get('categorical_crossentropy')(K.squeeze(K.one_hot(labels, 4208), 3), predictions), axis=[1,2]) / 1000.
      return K.squeeze(K.gradients(loss, [trainable_params]), 0) + kld_gradients
    else:
      loss = K.sum(losses.get(self.wrapper.loss)(labels, predictions), axis=[1,2]) / 1000.
      return K.stop_gradient(K.squeeze(K.gradients(loss, [trainable_params]), 0)) + kld_gradients

  def compute_kld_gradients(self, trainable_params, predictions):
    if not self.use_kld_regularization:
      return 0

    kld = K.mean(losses.get('kld')(K.stop_gradient(self.original_predictions), predictions), axis=[1,2])
    if self.use_second_order_derivatives:
      return self.kld_weight * K.squeeze(K.gradients(kld, [trainable_params]), 0)
    else:
      return self.kld_weight * K.stop_gradient(K.squeeze(K.gradients(kld, [trainable_params]), 0))

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
      'use_lr_per_step': self.use_lr_per_step,
      'use_kld_regularization': self.use_kld_regularization,
      'train_params': self.train_params,
      'correctly_serialized': self.correctly_serialized,
    }

  @classmethod
  def from_config(cls, config, custom_objects=None):
    wrapper = deserialize(config.pop('wrapper'), custom_objects=custom_objects)
    use_second_order_derivatives = config.get('use_second_order_derivatives', False)
    use_lr_per_step = config.get('use_lr_per_step', False)
    use_kld_regularization = config.get('use_kld_regularization', False)
    train_params = config.get('train_params', True)
    correctly_serialized = config.get('correctly_serialized', False)

    return cls(wrapper, config.get('num_steps', 3), use_second_order_derivatives, use_lr_per_step, use_kld_regularization, train_params, correctly_serialized=correctly_serialized)

  @property
  def trainable_weights(self):
    if not self.wrapper.built:
      self.wrapper.build(None)

    if self.correctly_serialized:
      return self._trainable_weights + self.wrapper.trainable_weights
    else:
      return self._trainable_weights

  @property
  def non_trainable_weights(self):
    if not self.wrapper.built:
      self.wrapper.build(None)

    if self.correctly_serialized:
      return self._non_trainable_weights + self.wrapper.non_trainable_weights
    else:
      return self._non_trainable_weights
