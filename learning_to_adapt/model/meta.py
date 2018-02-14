from keras import backend as K
from keras import losses
from keras.engine import InputSpec
from keras.engine.topology import Layer
from keras.initializers import Ones, Zeros, Constant
from keras.layers import Input, GaussianNoise, deserialize
from keras.models import Model, load_model

from loop import rnn
from wrapper import ModelWrapper, create_model_wrapper


def create_meta_learner(wrapper, units=20, meta_learner_type='full', input_type='frames'):
  feat_dim = wrapper.feat_dim
  num_params = wrapper.num_params

  if input_type == 'frames':
    training_feats = Input(shape=(None, None, feat_dim,))
    training_labels = Input(shape=(None, None, 1,))
    testing_feats = Input(shape=(None, feat_dim,))
  elif input_type == 'sequences':
    training_feats = Input(shape=(None, None, None, feat_dim,))
    training_labels = Input(shape=(None, None, None, 1,))
    testing_feats = Input(shape=(None, None, feat_dim,))
  else:
    raise ValueError('Undefined input type: %s' % input_type)

  params = Input(shape=(num_params,))

  if meta_learner_type == 'lr_per_parameter':
    meta_learner = LearningRatePerParameterMetaLearner(wrapper)
  elif meta_learner_type == 'full':
    meta_learner = MetaLearner(wrapper, units)
  else:
    raise ValueError('Undefined meta learner type: %s' % meta_learner_type)

  new_params = meta_learner([training_feats, training_labels, params])
  predictions = wrapper([new_params, testing_feats])

  return Model(
    inputs=[params, training_feats, training_labels, testing_feats],
    outputs=[predictions]
  )


def load_meta_learner(model, path):
  custom_objects={
    'MetaLearner': MetaLearner,
    'ModelWrapper': ModelWrapper,
    'LearningRatePerParameterMetaLearner': LearningRatePerParameterMetaLearner
  }

  model = load_model(path, custom_objects=custom_objects)
  inputs = model.inputs[:3]
  outputs = [model.layers[-3].output]

  return Model(inputs=inputs, outputs=outputs)


class MetaLearner(Layer):

  def __init__(self, wrapper, units, **kwargs):
    super(MetaLearner, self).__init__(**kwargs)

    self.wrapper = wrapper
    self.param_groups = list(wrapper.param_groups())
    self.num_param_groups = len(self.param_groups)
    self.input_dim = 5
    self.units = units

  def build(self, input_shapes):
    self.kernel = self.add_weight(
        shape=(self.num_param_groups, self.input_dim, self.units * 4),
        name='kernel',
        initializer='glorot_uniform',
    )

    self.recurrent_kernel = self.add_weight(
        shape=(self.num_param_groups, self.units, self.units * 4),
        name='recurrent_kernel',
        initializer='orthogonal',
    )

    self.bias = self.add_weight(
        shape=(self.num_param_groups, self.units * 4),
        name='bias',
        initializer=self.bias_initializer,
    )

    self.W_f = self.add_weight(
        shape=(self.num_param_groups, self.units + 1, 1),
        name='W_f',
        initializer='glorot_uniform',
    )

    self.b_f = self.add_weight(
        shape=(self.num_param_groups, 1),
        name='b_f',
        initializer='ones',
    )

    self.W_i = self.add_weight(
        shape=(self.num_param_groups, self.units + 1, 1),
        name='W_i',
        initializer='glorot_uniform',
    )

    self.b_i = self.add_weight(
        shape=(self.num_param_groups, 1),
        name='b_i',
        initializer='zeros',
    )

  def bias_initializer(self, shape, *args, **kwargs):
    return K.concatenate([
        Zeros()((self.num_param_groups, self.units), *args, **kwargs),
        Ones()((self.num_param_groups, self.units), *args, **kwargs),
        Zeros()((self.num_param_groups, self.units * 2), *args, **kwargs),
    ])

  def call(self, inputs):
    feats, labels, params = inputs

    self.params = params
    trainable_params = self.wrapper.get_trainable_params(params)

    last_output, _, _ = rnn(
      step_function=self.step,
      inputs=[feats, labels],
      initial_states=self.get_initital_state(trainable_params),
    )

    new_params = K.reshape(last_output[0], (1, self.wrapper.num_trainable_params))
    return self.wrapper.merge_params(self.params, new_params)

  def get_initital_state(self, params):
    return  [
        K.zeros((self.wrapper.num_trainable_params, self.units)),
        K.zeros((self.wrapper.num_trainable_params, self.units)),
        K.reshape(params, (self.wrapper.num_trainable_params, 1)),
        K.zeros((self.wrapper.num_trainable_params, 1)),
        K.zeros((self.wrapper.num_trainable_params, 1)),
    ]

  def compute_output_shape(self, input_shape):
    return input_shape[2]

  def step(self, inputs, states):
    feats, labels = inputs
    h_prev, c_prev, params, f_prev, i_prev = tuple(states)

    inputs, gradients = self.compute_inputs(params, feats, labels)

    h = []
    c = []
    new_params = []
    f = []
    i = []
    for param_group, indices in enumerate(self.param_groups):
      s, e = indices
      h_, c_ = self.lstm_step(inputs[s:e], h_prev[s:e], c_prev[s:e], param_group)
      new_params_, f_, i_ = self.update_params(params[s:e], gradients[s:e], h_, f_prev[s:e], i_prev[s:e], param_group)

      h.append(h_)
      c.append(c_)
      new_params.append(new_params_)
      f.append(f_)
      i.append(i_)

    return self.concatenate_all([new_params]), self.concatenate_all([h, c, new_params, f, i])

  def concatenate_all(self, xs):
    return [K.concatenate(x, axis=0) for x in xs]

  def compute_inputs(self, trainable_params, feats, labels):
    predictions = self.wrapper([self.params, K.transpose(trainable_params), feats])
    loss = K.mean(losses.get(self.wrapper.loss)(labels, predictions))
    gradients = K.stop_gradient(K.squeeze(K.gradients(loss, [trainable_params]), 0))

    loss = loss * K.ones_like(trainable_params)
    preprocessed_gradients = self.preprocess(gradients)
    preprocessed_loss = self.preprocess(loss)

    inputs = K.stop_gradient(K.concatenate([
      trainable_params,
      preprocessed_gradients[0],
      preprocessed_gradients[1],
      preprocessed_loss[0],
      preprocessed_loss[1],
    ], axis=1))

    return inputs, gradients

  def lstm_step(self, inputs, h, c, param_group):
    z = K.dot(inputs, self.kernel[param_group])
    z += K.dot(h, self.recurrent_kernel[param_group])
    z = K.bias_add(z, self.bias[param_group])

    z0 = z[:, :self.units]
    z1 = z[:, self.units: 2 * self.units]
    z2 = z[:, 2 * self.units: 3 * self.units]
    z3 = z[:, 3 * self.units:]

    i = K.hard_sigmoid(z0)
    f = K.hard_sigmoid(z1)
    c = f * c + i * K.tanh(z2)
    o = K.hard_sigmoid(z3)

    h = o * K.tanh(c)
    return h, c

  def update_params(self, params, gradients, h, f, i, param_group):
    f = K.relu(K.dot(K.concatenate([h, f], axis=1), self.W_f[param_group]) + self.b_f[param_group])
    i = K.relu(K.dot(K.concatenate([h, i], axis=1), self.W_i[param_group]) + self.b_i[param_group])
    new_params = f * params - i * gradients

    return new_params, f, i

  def preprocess(self, x):
    P = 10.0
    expP = K.cast(K.exp(P), K.floatx())
    negExpP = K.cast(K.exp(-P), K.floatx())

    m1 = K.cast(K.greater(K.abs(x), negExpP), K.floatx())
    m2 = K.cast(K.less_equal(K.abs(x), negExpP), K.floatx())

    return (
        m1 * K.log(K.abs(x) + m2) / P - m2,
        m1 * K.sign(x) + m2 * expP * x
    )

  def get_config(self):
    return {
      'units': self.units,
      'wrapper': {
        'class_name': self.wrapper.__class__.__name__,
        'config': self.wrapper.get_config()
      }
    }

  @classmethod
  def from_config(cls, config, custom_objects=None):
    units = config.pop('units')
    wrapper = deserialize(config.pop('wrapper'), custom_objects=custom_objects)

    return cls(wrapper, units)


class LearningRatePerParameterMetaLearner(Layer):

  def __init__(self, wrapper, **kwargs):
    super(LearningRatePerParameterMetaLearner, self).__init__(**kwargs)

    self.wrapper = wrapper
    self.num_trainable_params = self.wrapper.num_trainable_params

  def build(self, input_shapes):
    self.learning_rate = self.add_weight(
      shape=(self.num_trainable_params, 1),
      name='learning_rate',
      initializer='zeros'
    )

  def call(self, inputs):
    feats, labels, params = inputs

    self.params = params
    trainable_params = self.wrapper.get_trainable_params(params)

    last_output, _, _ = rnn(
      step_function=self.step,
      inputs=[feats, labels],
      initial_states=self.get_initital_state(trainable_params),
    )

    new_params = K.reshape(last_output[0], (1, self.wrapper.num_trainable_params))
    return self.wrapper.merge_params(self.params, new_params)

  def get_initital_state(self, params):
    return  [K.reshape(params, (self.num_trainable_params, 1))]

  def compute_output_shape(self, input_shape):
    return input_shape[2]

  def step(self, inputs, states):
    feats, labels = inputs
    params = states[0]
    gradients = self.compute_gradients(params, feats, labels)
    new_params = params - K.exp(self.learning_rate) * gradients

    return [new_params], [new_params]

  def compute_gradients(self, trainable_params, feats, labels):
    predictions = self.wrapper([self.params, K.transpose(trainable_params), feats])
    loss = K.mean(losses.get(self.wrapper.loss)(labels, predictions))
    return K.stop_gradient(K.squeeze(K.gradients(loss, [trainable_params]), 0))

  def get_config(self):
    return {
      'wrapper': {
        'class_name': self.wrapper.__class__.__name__,
        'config': self.wrapper.get_config()
      }
    }

  @classmethod
  def from_config(cls, config, custom_objects=None):
    wrapper = deserialize(config.pop('wrapper'), custom_objects=custom_objects)

    return cls(wrapper)
