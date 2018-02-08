import re
import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D, Activation


def parse_nnet3(path):
    with open(path, 'r') as f:
        line = f.readline().strip()
        assert line == "<Nnet3>"

        definition = parse_node_definitions(f)
        components = parse_components(f)

        line = f.readline().strip()
        assert line == "</Nnet3>"  

        return definition, components


def parse_node_definitions(f):
    nodes = {}

    while True:
        line = f.readline().strip()

        if line == "":
            break

        if line.startswith("input-node"):
            match = re.match("^input-node name=(.*?) dim=(.*?)$", line)
            nodes[match.group(1)] = {
                "type": "input",
                "dim": match.group(2),
                "input": None
            }

        if line.startswith("component-node"):
            match = re.match("^component-node name=(.*?) component=(.*?) input=(.*?)$", line)

            if match.group(2).endswith("affine"):
                input_node, kernel_size, dilation_rate, offsets = parse_offsets(match.group(3))

                nodes[match.group(1)] = {
                    "type": "affine",
                    "name": match.group(2),
                    "kernel_size": kernel_size,
                    "strides": 1, 
                    "dilation_rate": dilation_rate,
                    "input": input_node,
                    "offsets": offsets
                }
            elif match.group(2).endswith("relu"):
                nodes[match.group(1)] = {
                    "type": "activation",
                    "activation": "relu",
                    "input": match.group(3)
                }
            elif match.group(2).endswith("log-softmax"):
                nodes[match.group(1)] = {
                    "type": "activation",
                    "activation": "softmax",
                    "input": match.group(3)
                }
            else:
                raise ValueError("Unknown component: %s" % match.group(2))

        if line.startswith("output-node"):
            match = re.match("^output-node name=(.*?) input=(.*?) objective=(.*?)$", line)
            nodes[match.group(1)] = {
                "type": "activation",
                "input": match.group(2),
                "activation": match.group(3)
            }

    return nodes


def parse_offsets(inputs):
    if inputs.startswith("Append"):
        offsets = []
        inputs = inputs[7:-1]

        while inputs:
            if inputs.startswith("Offset"):
                current_input = inputs[7:inputs.index(")")].split(",")
                input_node = current_input[0]
                offsets.append(int(current_input[1]))
                inputs = inputs[inputs.index(")") + 3:]
            else:
                if inputs.find(",") != -1:
                    input_node = inputs[:inputs.index(",")]
                    offsets.append(0)
                    inputs = inputs[inputs.index(",") + 2:]
                else:
                    input_node = inputs
                    offsets.append(0)
                    inputs = ""
    else:
        input_node = inputs
        offsets = [0]

    kernel_size = len(offsets)

    gaps = [y-x for (x,y) in zip(offsets[:-1], offsets[1:])]
    if not gaps:
        dilation_rate = 1
    elif all([x == gaps[0] for x in gaps]):
        dilation_rate = gaps[0]
    else:
        raise ValueError("Following TDNN splice is not supported.: %s" % ", ".join(offsets))

    return input_node, kernel_size, dilation_rate, offsets


def parse_components(f):
    line = f.readline().strip()
    assert line.startswith("<NumComponents>")
    num_components = int(line.split()[1])

    components = {}
    while num_components:
        line = f.readline().strip()

        if line.startswith("<ComponentName>"):
            num_components -= 1

            match = re.match("<ComponentName> (.*?.(relu|log-softmax)) .* <Dim> (.*?) ", line)
            if match:
                continue

            match = re.match("<ComponentName> (.*?.affine) ", line)
            if match:
                matrix = []
                while True:
                    line = f.readline().strip()
                    matrix.append(np.array([float(x) for x in line.strip("[]").split()], dtype="float32"))

                    if line.endswith("]"):
                        break

                matrix = np.stack(matrix)
                line = f.readline().strip()
                assert line.startswith("<BiasParams>")
                bias = np.array([float(x) for x in line[16:-1].split()], dtype="float32")

                components[match.group(1)] = (matrix, bias)

                line = f.readline().strip()
                continue

    return components


def create_model(definition, components, subsampling_factor):
    graph = []
    node_name = "output"
    while node_name:
        current_node = definition[node_name]

        if current_node["type"] == "affine" and subsampling_factor > 1:
            if current_node["dilation_rate"] == subsampling_factor or current_node["kernel_size"] == 1:
                current_node["dilation_rate"] = 1
            elif current_node["dilation_rate"] == 1:
                current_node["strides"] = subsampling_factor
                subsampling_factor = 1 
            else:
                raise ValueError("Unable to optimize graph.")

        graph.append((node_name, current_node))
        node_name = current_node["input"]

    model = Sequential()
    left_context = 0
    right_context = 0
    for node_name, definition in reversed(graph):
        if definition["type"] == "input":
            size = int(definition["dim"])
        elif definition["type"] == "activation":
            model.add(Activation(definition["activation"], name=node_name))
        elif definition["type"] == "affine":
            weights = components[definition["name"]]
            filters = weights[1].shape[0]
            kernel_size = definition["kernel_size"]
            strides = definition["strides"]
            dilation_rate = definition["dilation_rate"]
            padding = "valid"
            left_context += definition["offsets"][0]
            right_context += definition["offsets"][-1]

            kernel = weights[0].T.reshape((kernel_size, size, filters))
            bias = weights[1]

            model.add(Conv1D(
                filters, kernel_size, strides=strides, padding=padding,
                dilation_rate=dilation_rate, use_bias=True,
                weights=(kernel, bias), input_shape=(None, size),
		name = node_name
            ))

            size = filters

    return model, left_context, right_context


if __name__ == "__main__":
    workdir = sys.argv[1]
    subsampling_factor = int(sys.argv[2])

    input_path = "%s/final.txt" % workdir
    output_path = "%s/dnn.nnet.h5" % workdir

    definition, components = parse_nnet3(input_path)
    model, left_context, right_context = create_model(definition, components, subsampling_factor)

    model.summary()
    model.save(output_path)

    with open("%s/context_opts" % workdir, "w") as f:
        print >> f, left_context, right_context

