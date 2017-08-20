import keras.backend as K
from keras.models import Model
from keras.layers import Input
from keras.engine.topology import Layer
import numpy as np
import unittest

from loop import rnn


class TestLoop(unittest.TestCase):

  def testLoop(self):
    input_x = Input(shape=(5,1,))
    input_y = Input(shape=(5,1,))
    initial_state = Input(shape=(1,))
    output = RNNStub()([input_x, input_y, initial_state])
    model = Model(inputs=[input_x, input_y, initial_state], outputs=[output])

    x_ = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).reshape((1, 5, 1))
    y_ = np.array([6.0, 7.0, 8.0, 9.0, 10.0]).reshape((1, 5, 1))
    initial_state_ = np.zeros((1,))

    prediction = model.predict([x_, y_, initial_state_])
    self.assertEqual(55, prediction)


class RNNStub(Layer):

  def call(self, inputs):
    def step(inputs, state):
      new_state = inputs[0] + inputs[1] + state[0]
      return [new_state], [new_state]

    output, _, _ = rnn(step, [inputs[0], inputs[1]], [inputs[2]])

    return output
