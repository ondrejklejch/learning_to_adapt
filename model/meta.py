from keras import backend as K
from keras import losses
from keras.engine import InputSpec
from keras.engine.topology import Layer
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
  training_labels = Input(shape=(None, None, num_labels,))
  testing_feats = Input(shape=(None, feat_dim,))
  params = Input(shape=(num_params,))

  meta_learner = MetaLearner(wrapper, LSTM(units, implementation=2))
  new_params = meta_learner([training_feats, training_labels, params])
  predictions = wrapper([new_params, testing_feats])

  return Model(
    inputs=[params, training_feats, training_labels, testing_feats],
    outputs=[predictions]
  )


class MetaLearner(Layer):

  def __init__(self, wrapper, hidden_layer, **kwargs):
    super(MetaLearner, self).__init__(**kwargs)

    self.wrapper = wrapper
    self.hidden_layer = hidden_layer
    self.input_spec = [InputSpec(ndim=4), InputSpec(ndim=4), InputSpec(ndim=2)]

  def build(self, input_shapes):
    self.hidden_layer.build((None, None, 5))

    self.W_f = self.add_weight(
        shape=(self.hidden_layer.units + 1, 1),
        name='W_f',
        initializer='glorot_uniform',
    )

    self.b_f = self.add_weight(
        shape=(1,),
        name='b_f',
        initializer='ones',
    )

    self.W_i = self.add_weight(
        shape=(self.hidden_layer.units + 1, 1),
        name='W_i',
        initializer='glorot_uniform',
    )

    self.b_i = self.add_weight(
        shape=(1,),
        name='b_i',
        initializer='zeros',
    )

  def call(self, inputs):
    feats, labels, params = inputs
    initial_states =  [
        K.stack([K.zeros_like(params)] * self.hidden_layer.units, axis=2),
        K.stack([K.zeros_like(params)] * self.hidden_layer.units, axis=2),
        params,
        K.zeros_like(params),
        K.zeros_like(params)
    ]

    last_output, _, _ = rnn(
      step_function=self.step,
      inputs=[feats, labels],
      initial_states=initial_states,
    )

    return last_output[0]

  def compute_output_shape(self, input_shape):
    return input_shape[2]

  def step(self, inputs, states):
    feats, labels = inputs
    h_prev, c_prev, params, f_prev, i_prev = tuple(states)

    predictions = self.wrapper([params, feats])
    loss = K.sum(losses.get(self.wrapper.model.loss)(labels, predictions), axis=1)
    gradients = K.stop_gradient(K.gradients(loss, [params]))
    gradients = K.squeeze(K.gradients(loss, [params]), 0)

    loss = K.repeat_elements(K.expand_dims(loss, 1), K.int_shape(params)[-1], axis=1)
    preprocessed_gradients = self.preprocess(gradients)
    preprocessed_loss = self.preprocess(loss)

    meta_learner_inputs = K.stop_gradient(K.stack([
      params,
      preprocessed_gradients[0],
      preprocessed_gradients[1],
      preprocessed_loss[0],
      preprocessed_loss[1],
    ], axis=2))

    params_shape = K.shape(params)
    h_shape = K.shape(h_prev)

    params = K.reshape(params, (-1, 1))
    gradients = K.reshape(gradients, (-1, 1))
    f_prev = K.reshape(f_prev, (-1, 1))
    i_prev = K.reshape(i_prev, (-1, 1))
    h_prev = K.reshape(h_prev, (-1, self.hidden_layer.units))
    c_prev = K.reshape(c_prev, (-1, self.hidden_layer.units))
    meta_learner_inputs = K.reshape(meta_learner_inputs, (-1, K.int_shape(meta_learner_inputs)[-1]))

    constants = self.hidden_layer.get_constants(meta_learner_inputs, training=None)
    _, (h, c) = self.hidden_layer.step(meta_learner_inputs, (h_prev, c_prev) + tuple(constants))
    f = K.sigmoid(K.dot(K.concatenate([h, f_prev], axis=1), self.W_f) + self.b_f)
    i = K.sigmoid(K.dot(K.concatenate([h, i_prev], axis=1), self.W_i) + self.b_i)
    new_params = f * params - i * gradients

    new_params = K.reshape(new_params, params_shape)
    f = K.reshape(f, params_shape)
    i = K.reshape(i, params_shape)
    h = K.reshape(h, h_shape)
    c = K.reshape(c, h_shape)

    return [new_params], [h, c, new_params, f, i]

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
