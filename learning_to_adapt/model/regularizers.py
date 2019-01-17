import keras.backend as K
from keras.regularizers import Regularizer


class L0(Regularizer):

    def __init__(self, l0, beta, gamma, delta):
        self.l0 = l0
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def __call__(self, loga):
        l0 = self.l0 * K.sum(K.sigmoid(loga - self.beta * K.log(-self.gamma / self.delta)))

        if K.ndim(loga) > 1:
            return l0 / K.cast(K.shape(loga)[0], K.floatx())
        else:
            return l0

    def get_config(self):
        return {
            'l0': self.l0,
            'beta': self.beta,
            'gamma': self.gamma,
            'delta': self.delta
        }
