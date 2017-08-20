import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import control_flow_ops
import keras.backend as K


def rnn(step_function, inputs, initial_states,
    go_backwards=False, mask=None, constants=None,
    unroll=False, input_length=None):
  states, constants = prepare_states(initial_states, constants)
  inputs = prepare_inputs(inputs)
  time, time_steps, inputs_ta, outputs_ta = prepare_tensors(step_function, inputs, initial_states, constants)
  final_outputs = run(step_function, time, time_steps, inputs_ta, outputs_ta, states, constants)

  return process_outputs(final_outputs)

def run(step_function, time, time_steps, inputs_ta, outputs_ta, states, constants):
  def _step(time, outputs_ta_t, *states):
    """RNN step function.

    # Arguments
      time: Current timestep value.
      output_ta_t: TensorArray.
      *states: List of states.

    # Returns
      Tuple: `(time + 1,output_ta_t) + tuple(new_states)`
    """
    current_inputs = [input_ta.read(time) for input_ta in inputs_ta]
    outputs, new_states = step_function(current_inputs, tuple(states) + tuple(constants))
    for state, new_state in zip(states, new_states):
      new_state.set_shape(state.get_shape())

    for i, output in enumerate(outputs):
      outputs_ta_t[i] = outputs_ta_t[i].write(time, output)

    return (time + 1, outputs_ta_t) + tuple(new_states)

  final_outputs = control_flow_ops.while_loop(
    cond=lambda time, *_: time < time_steps,
    body=_step,
    loop_vars=(time, outputs_ta) + states,
    parallel_iterations=32,
    swap_memory=True)

  return final_outputs

def prepare_states(initial_states, constants):
  if constants is None:
    constants = []

  return tuple(initial_states), constants

def prepare_inputs(inputs_list):
  if not isinstance(inputs_list, list):
    raise ValueError('Inputs should be a list of tensors.')

  new_inputs = []
  for inputs in inputs_list:
    ndim = len(inputs.get_shape())
    if ndim < 3:
      raise ValueError('Input should be at least 3D.')
    axes = [1, 0] + list(range(2, ndim))
    inputs = tf.transpose(inputs, (axes))
    new_inputs.append(inputs)

  return new_inputs

def prepare_tensors(step_function, inputs_list, initial_states, constants):
  time = tf.constant(0, dtype='int32', name='time')
  time_steps = tf.shape(inputs_list[0])[0]
  outputs_list, _ = step_function([x[0] for x in inputs_list], tuple(initial_states) + tuple(constants))

  outputs_ta = []
  for i, outputs in enumerate(outputs_list):
    outputs_ta.append(tensor_array_ops.TensorArray(
      dtype=outputs.dtype,
      size=time_steps,
      tensor_array_name='output_ta_%d' % i))

  inputs_ta = []
  for i, inputs in enumerate(inputs_list):
    input_ta = tensor_array_ops.TensorArray(
      dtype=inputs.dtype,
      size=time_steps,
      tensor_array_name='input_ta_%d' % i)

    inputs_ta.append(input_ta.unstack(inputs))

  return time, time_steps, inputs_ta, outputs_ta

def process_outputs(final_outputs):
  last_time = final_outputs[0]
  outputs_ta = final_outputs[1]
  new_states = final_outputs[2:]

  outputs = [output_ta.stack() for output_ta in outputs_ta]
  last_outputs = [output_ta.read(last_time - 1) for output_ta in outputs_ta]

  for i in range(len(outputs)):
    axes = [1, 0] + list(range(2, len(outputs[i].get_shape())))
    outputs[i] = tf.transpose(outputs[i], axes)

  return last_outputs, outputs, new_states
