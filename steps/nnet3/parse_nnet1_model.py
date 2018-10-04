import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D, Activation
from learning_to_adapt.model.layers import LHUC, FeatureTransform


def parse_nnet1(line_buffer, with_lhuc_layers=False):
  components = []
  line = next(line_buffer)
  assert line.startswith("<Nnet>")

  for line in line_buffer:
    if line.startswith("</Nnet>"):
      break

    components.extend(parse_component(line, f, with_lhuc_layers))

  return components


def parse_feature_transform(line_buffer):
  for line in line_buffer:
    if line.startswith("</Nnet>"):
      break

    if line.startswith("<Rescale>"):
      line = " ".join(next(line_buffer).split()[2:])
      rescale = parse_vector(line)

    if line.startswith("<AddShift>"):
      line = " ".join(next(line_buffer).split()[2:])
      shift = parse_vector(line)

  return [
    FeatureTransform(input_shape=(None, rescale.shape[0]), weights=[rescale, shift])
  ]


def parse_component(line, line_buffer, with_lhuc_layers):
  if line.startswith("<AffineTransform>"):
    (_, output_dim, input_dim) = line.split()
    output_dim = int(output_dim)
    input_dim = int(input_dim)

    # Reads Learning Rate
    next(f)

    kernel = np.expand_dims(parse_weights(f, input_dim, output_dim).T, 0)
    bias = parse_bias(f, output_dim)

    parse_end_of_component(f)

    return [Conv1D(output_dim, 1, input_shape=(None, input_dim), weights=[kernel, bias], trainable=True)]
  elif line.startswith("<Sigmoid>"):
    parse_end_of_component(f)

    if with_lhuc_layers:
      return [Activation("sigmoid"), LHUC()]
    else:
      return [Activation("sigmoid")]
  elif line.startswith("<Softmax>"):
    parse_end_of_component(f)
    return [Activation("softmax")]
  else:
    print("ERROR", line)


def parse_weights(line_buffer, input_dim, output_dim):
  weights = np.zeros((output_dim, input_dim))

  i = 0
  while True:
    line = next(line_buffer)

    if line.strip().startswith("["):
      continue
    elif line.strip().endswith("]"):
      weights[i] = parse_vector(line)
      break
    else:
      weights[i] = parse_vector(line)
      i += 1

  return weights


def parse_bias(line_buffer, output_dim):
  return parse_vector(next(line_buffer))


def parse_vector(line, number_type=float, dtype="float32"):
  vector = line.strip().strip("[]")
  return np.array([number_type(x) for x in vector.split()], dtype=dtype)


def parse_end_of_component(line_buffer):
  line = next(line_buffer)
  assert line.startswith("<!EndOfComponent>")


if __name__ == "__main__":
  components = []

  root = sys.argv[1]
  feature_transform = "%s/final.feature_transform" % root
  model = "%s/final.txt" % root
  output = root

  with open(feature_transform, "r") as f:
    components.extend(parse_feature_transform(f))

  with open(model, "r") as f:
    components.extend(parse_nnet1(f, with_lhuc_layers=True))

  model = Sequential()
  for component in components:
    model.add(component)

  model.save("%s/dnn.nnet.h5" % output)
  model.summary()
