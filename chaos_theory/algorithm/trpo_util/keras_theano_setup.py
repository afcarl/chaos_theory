import copy_reg, cPickle
import numpy as np
import theano, keras, keras.backend

floatX = theano.config.floatX  # pylint: disable=E1101
keras.backend.set_floatx(floatX)
keras.backend.set_epsilon(1e-7)

FNOPTS = dict(allow_input_downcast=True, on_unused_input='ignore')

from keras.models import Sequential, model_from_json
from keras.layers.core import Layer


class ConcatFixedStd(Layer):

    input_ndim = 2

    def __init__(self, **kwargs):
        Layer.__init__(self, **kwargs)

    def build(self, input_shape):
        input_dim = input_shape[1]
        self.logstd = theano.shared(np.zeros(input_dim,floatX), name='{}_logstd'.format(self.name))
        self.trainable_weights = [self.logstd]

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1] * 2)

    def call(self, x, mask):
        Mean = x
        Std = T.repeat(T.exp(self.logstd)[None, :], Mean.shape[0], axis=0)
        return T.concatenate([Mean, Std], axis=1)

def kerasmodel_unpickler(s):
    print "unpickling keras model"
    modelstr, weightss = cPickle.loads(s)
    model = model_from_json(modelstr, custom_objects={"ConcatFixedStd": ConcatFixedStd})
    assert len(model.layers) == len(weightss)
    for (layer, weights) in zip(model.layers, weightss):
        layer.set_weights(weights)
    return model


def kerasmodel_pickler(model):
    print "pickling keras model"
    modelstr = model.to_json()
    weightss = []
    for layer in model.layers:
        weightss.append(layer.get_weights())
    s = cPickle.dumps((modelstr, weightss), -1)
    return kerasmodel_unpickler, (s,)


def function_pickler(_):
    raise RuntimeError("Trying to pickle theano function")

copy_reg.pickle(Sequential, kerasmodel_pickler, kerasmodel_unpickler)
copy_reg.pickle(theano.compile.function_module.Function, function_pickler)