import numpy as np

def load_lda(path):
    rows = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip(" []\n")

            if line:
                rows.append(np.fromstring(line, dtype=np.float32, sep=' '))

    matrix = np.array(rows).T

    return matrix[:-1], matrix[-1]


