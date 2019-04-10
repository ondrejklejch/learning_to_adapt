from keras.models import load_model as keras_load_model

from layers import *
from meta import *
from maml import *
from regularizers import *
from wrapper import *

def load_model(path, adaptation_type=None):
    model = keras_load_model(path, custom_objects={
        'FeatureTransform': FeatureTransform,
        'LHUC': LHUC,
        'SparseLHUC': SparseLHUC,
        'Renorm': Renorm,
        'Multiply': Multiply,
        'SparseMultiply': SparseMultiply,
        'SDBatchNormalization': SDBatchNormalization,
        'UttBatchNormalization': UttBatchNormalization,
        'LDA': LDA,
        'L0': L0,
        'MetaLearner': MetaLearner,
        'LearningRatePerLayerMetaLearner': LearningRatePerLayerMetaLearner,
        'MAML': MAML,
        'ModelWrapper': ModelWrapper,
    })

    if adaptation_type == "LHUC":
        for l in model.layers:
            l.trainable = l.name.startswith("lhuc")
    elif adaptation_type == "BATCHNORM":
        for l in model.layers:
            l.trainable = l.name.endswith("batchnorm")
    elif adaptation_type == "ALL":
        for l in model.layers:
            l.trainable = l.name != "lda" and not l.name.startswith("lhuc")

    return model
