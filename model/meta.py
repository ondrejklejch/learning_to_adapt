from keras import backend as K
from keras import losses
from keras.engine import InputSpec
from keras.engine.topology import Layer
from keras.initializers import Ones, Zeros
from keras.layers import Activation, Dense, Input, LSTM, Recurrent
from keras.models import Model

from loop import rnn
from wrapper import ModelWrapper


def create_meta_learner(model, units=20):
  wrapper = ModelWrapper(model)
  feat_dim = wrapper.model.layers[0].input_shape[-1]
  num_labels = wrapper.model.layers[-1].output_shape[-1]
  num_params = len(wrapper.get_all_weights())

  training_feats = Input(shape=(None, None, feat_dim,))
  training_labels = Input(shape=(None, None, 1,))
  testing_feats = Input(shape=(None, feat_dim,))
  params = Input(shape=(num_params,))

  meta_learner = MetaLearner(wrapper, units)
  new_params = meta_learner([training_feats, training_labels, params])
  predictions = wrapper([new_params, testing_feats])

  return Model(
    inputs=[params, training_feats, training_labels, testing_feats],
    outputs=[predictions]
  )


class MetaLearner(Layer):

  def __init__(self, wrapper, units, **kwargs):
    super(MetaLearner, self).__init__(**kwargs)

    self.wrapper = wrapper
    self.num_params = len(wrapper.get_all_weights())
    self.input_dim = 5
    self.units = units
    self.input_spec = [InputSpec(ndim=4), InputSpec(ndim=4), InputSpec(ndim=2)]

  def build(self, input_shapes):
    self.kernel = self.add_weight(
        shape=(self.input_dim, self.units * 4),
        name='kernel',
        initializer='glorot_uniform',
    )

    self.recurrent_kernel = self.add_weight(
        shape=(self.units, self.units * 4),
        name='recurrent_kernel',
        initializer='orthogonal',
    )

    self.bias = self.add_weight(
        shape=(self.units * 4,),
        name='bias',
        initializer=self.bias_initializer,
    )

    self.W_f = self.add_weight(
        shape=(self.units + 1, 1),
        name='W_f',
        initializer='glorot_uniform',
    )

    self.b_f = self.add_weight(
        shape=(1,),
        name='b_f',
        initializer='ones',
    )

    self.W_i = self.add_weight(
        shape=(self.units + 1, 1),
        name='W_i',
        initializer='glorot_uniform',
    )

    self.b_i = self.add_weight(
        shape=(1,),
        name='b_i',
        initializer='zeros',
    )

  def bias_initializer(self, shape, *args, **kwargs):
    return K.concatenate([
        Zeros()((self.units,), *args, **kwargs),
        Ones()((self.units,), *args, **kwargs),
        Zeros()((self.units * 2,), *args, **kwargs),
    ])

  def call(self, inputs):
    feats, labels, params = inputs
    last_output, _, _ = rnn(
      step_function=self.step,
      inputs=[feats, labels],
      initial_states=self.get_initital_state(params),
    )

    return K.reshape(last_output[0], (1, self.num_params))

  def get_initital_state(self, params):
    return  [
        K.zeros((self.num_params, self.units)),
        K.zeros((self.num_params, self.units)),
        K.reshape(params, (self.num_params, 1)),
        K.zeros((self.num_params, 1)),
        K.zeros((self.num_params, 1)),
    ]

  def compute_output_shape(self, input_shape):
    return input_shape[2]

  def step(self, inputs, states):
    feats, labels = inputs
    h_prev, c_prev, params, f_prev, i_prev = tuple(states)

    inputs, gradients = self.compute_inputs(params, feats, labels)
    h, c = self.lstm_step(inputs, h_prev, c_prev)
    new_params, f, i = self.update_params(params, gradients, h, f_prev, i_prev)

    return [new_params], [h, c, new_params, f, i]

  def compute_inputs(self, params, feats, labels):
    predictions = self.wrapper([K.transpose(params), feats])
    loss = K.sum(losses.get(self.wrapper.model.loss)(labels, predictions))
    gradients = K.stop_gradient(K.squeeze(K.gradients(loss, [params]), 0))

    loss = loss * K.ones_like(params)
    preprocessed_gradients = self.preprocess(gradients)
    preprocessed_loss = self.preprocess(loss)

    inputs = K.stop_gradient(K.concatenate([
      params,
      preprocessed_gradients[0],
      preprocessed_gradients[1],
      preprocessed_loss[0],
      preprocessed_loss[1],
    ], axis=1))

    return inputs, gradients

  def lstm_step(self, inputs, h, c):
    z = K.dot(inputs, self.kernel)
    z += K.dot(h, self.recurrent_kernel)
    z = K.bias_add(z, self.bias)

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

  def update_params(self, params, gradients, h, f, i):
    f = K.sigmoid(K.dot(K.concatenate([h, f], axis=1), self.W_f) + self.b_f)
    i = K.sigmoid(K.dot(K.concatenate([h, i], axis=1), self.W_i) + self.b_i)
    new_params = f * params - i * gradients

    return new_params, f, i

  def preprocess(self, x):
    P = 10.0
    expP = K.exp(P)
    negExpP = K.exp(-P)

    m1 = K.cast(K.greater(K.abs(x), negExpP), K.floatx())
    m2 = K.cast(K.less_equal(K.abs(x), negExpP), K.floatx())

    return (
        m1 * K.log(K.abs(x) + m2) / P - m2,
        m1 * K.sign(x) + m2 * expP * x
    )
