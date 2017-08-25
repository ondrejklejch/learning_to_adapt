from keras import backend as K
from keras import losses
from keras.engine import InputSpec
from keras.engine.topology import Layer
from keras.layers import Activation, Dense, Input, LSTM, Recurrent
from keras.models import Model

from loop import rnn
from wrapper import ModelWrapper


def create_meta_learner(model):
  wrapper = ModelWrapper(model)
  params = wrapper.get_all_weights()
  feat_dim = wrapper.model.layers[0].input_shape[-1]
  num_labels = wrapper.model.layers[-1].output_shape[-1]
  num_params = len(params)

  training_feats = Input(shape=(None, None, feat_dim,))
  training_labels = Input(shape=(None, None, num_labels,))
  testing_feats = Input(shape=(None, feat_dim,))
  params = Input(shape=(num_params,))

  meta_learner = MetaLearner(wrapper)
  new_params = meta_learner([training_feats, training_labels, params])
  predictions = wrapper([new_params, testing_feats])

  return Model(
    inputs=[params, training_feats, training_labels, testing_feats],
    outputs=[predictions]
  )


class MetaLearner(Layer):

  def __init__(self, wrapper, **kwargs):
    super(MetaLearner, self).__init__(**kwargs)

    self.wrapper = wrapper
    self.input_spec = [InputSpec(ndim=4), InputSpec(ndim=4), InputSpec(ndim=2)]

  def build(self, input_shapes):
    self.W_f = self.add_weight(
        shape=(6, 1),
        name='W_f',
        initializer='glorot_uniform',
    )

    self.b_f = self.add_weight(
        shape=(1,),
        name='b_f',
        initializer='ones',
    )

    self.W_i = self.add_weight(
        shape=(6, 1),
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
    initial_states =  [params, K.zeros_like(params), K.zeros_like(params)]

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
    params, f_prev, i_prev = tuple(states)

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

    new_params, f, i = self.compute_weights_update(params, gradients, meta_learner_inputs, f_prev, i_prev)

    return [new_params], [new_params, f, i]

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

  def compute_weights_update(self, params, gradients, h, f_prev, i_prev):
    params_shape = [x if x is not None else -1 for x in K.int_shape(params)]

    params = K.reshape(params, (-1, 1))
    gradients = K.reshape(gradients, (-1, 1))
    f_prev = K.reshape(f_prev, (-1, 1))
    i_prev = K.reshape(i_prev, (-1, 1))
    h = K.reshape(h, (-1, K.int_shape(h)[-1]))

    f = K.sigmoid(K.dot(K.concatenate([h, f_prev], axis=1), self.W_f) + self.b_f)
    i = K.sigmoid(K.dot(K.concatenate([h, i_prev], axis=1), self.W_i) + self.b_i)
    new_params = f * params - i * gradients

    new_params = K.reshape(new_params, params_shape)
    f = K.reshape(f, params_shape)
    i = K.reshape(i, params_shape)

    return new_params, f, i
