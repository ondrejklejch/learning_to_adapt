import keras.backend as K
from keras.regularizers import Regularizer


class L0(Regularizer):

    def __init__(self, l0, beta, gamma, delta):
        self.l0 = l0
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def __call__(self, loga):
        return self.l0 * K.sum(K.sigmoid(loga - self.beta * K.log(-self.gamma / self.delta)))

    def get_config(self):
        return {
            'l0': self.l0,
            'beta': self.beta,
            'gamma': self.gamma,
            'delta': self.delta
        }
