from keras.models import load_model as keras_load_model

from layers import *
from meta import *
from maml import *
from regularizers import *
from wrapper import *

def load_model(path):
    return keras_load_model(path, custom_objects={
        'FeatureTransform': FeatureTransform,
        'LHUC': LHUC,
        'SparseLHUC': SparseLHUC,
        'Renorm': Renorm,
        'Multiply': Multiply,
        'SparseMultiply': SparseMultiply,
        'SDBatchNormalization': SDBatchNormalization,
        'UttBatchNormalization': UttBatchNormalization,
        'L0': L0,
        'MAML': MAML,
        'ModelWrapper': ModelWrapper,
    })
