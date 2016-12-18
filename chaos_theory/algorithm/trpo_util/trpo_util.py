import scipy
import scipy.signal
import theano.tensor as T
import theano
import numpy as np

from .keras_theano_setup import FNOPTS, floatX

PG_OPTIONS = [
    ("timestep_limit", int, 100, "maximum length of trajectories"),
    ("n_iter", int, 200, "number of batch"),
    ("parallel", int, 0, "collect trajectories in parallel"),
    ("timesteps_per_batch", int, 10000, ""),
    ("gamma", float, 0.99, "discount"),
    ("lam", float, 1.0, "lambda parameter from generalized advantage estimation"),
]

class dict2(dict):
    "dictionary-like object that exposes its keys as attributes"
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self

def zipsame(*seqs):
    L = len(seqs[0])
    assert all(len(seq) == L for seq in seqs[1:])
    return zip(*seqs)

def flatten(arrs):
    return np.concatenate([arr.flat for arr in arrs])

def unflatten(vec, shapes):
    i=0
    arrs = []
    for shape in shapes:
        size = np.prod(shape)
        arr = vec[i:i+size].reshape(shape)
        arrs.append(arr)
        i += size
    return arrs

def flatgrad(loss, var_list):
    grads = T.grad(loss, var_list)
    return T.concatenate([g.flatten() for g in grads])

def update_default_config(tuples, usercfg):
    """
    inputs
    ------
    tuples: a sequence of 4-tuples (name, type, defaultvalue, description)
    usercfg: dict-like object specifying overrides
    outputs
    -------
    dict2 with updated configuration
    """
    out = dict2()
    for (name,_,defval,_) in tuples:
        out[name] = defval
    if usercfg:
        for (k,v) in usercfg.iteritems():
            if k in out:
                out[k] = v
    return out

class EzPickle(object):
    """Objects that are pickled and unpickled via their constructor
    arguments.
    Example usage:
        class Dog(Animal, EzPickle):
            def __init__(self, furcolor, tailkind="bushy"):
                Animal.__init__()
                EzPickle.__init__(furcolor, tailkind)
                ...
    When this object is unpickled, a new Dog will be constructed by passing the provided
    furcolor and tailkind into the constructor. However, philosophers are still not sure
    whether it is still the same dog.
    This is generally needed only for environments which wrap C/C++ code, such as MuJoCo
    and Atari.
    """
    def __init__(self, *args, **kwargs):
        self._ezpickle_args = args
        self._ezpickle_kwargs = kwargs
    def __getstate__(self):
        return {"_ezpickle_args" : self._ezpickle_args, "_ezpickle_kwargs": self._ezpickle_kwargs}
    def __setstate__(self, d):
        out = type(self)(*d["_ezpickle_args"], **d["_ezpickle_kwargs"])
        self.__dict__.update(out.__dict__)


class SetFromFlat(object):
    def __init__(self, var_list):
        theta = T.vector()
        start = 0
        updates = []
        for v in var_list:
            shape = v.shape
            size = T.prod(shape)
            updates.append((v, theta[start:start + size].reshape(shape)))
            start += size
        self.op = theano.function([theta], [], updates=updates, **FNOPTS)

    def __call__(self, theta):
        self.op(theta.astype(floatX))


class GetFlat(object):
    def __init__(self, var_list):
        self.op = theano.function([], T.concatenate([v.flatten() for v in var_list]), **FNOPTS)

    def __call__(self):
        return self.op()  # pylint: disable=E1101

class EzFlat(object):
    def __init__(self, var_list):
        self.gf = GetFlat(var_list)
        self.sff = SetFromFlat(var_list)
    def set_params_flat(self, theta):
        self.sff(theta)
    def get_params_flat(self):
        return self.gf()

def comma_sep_ints(s):
    if s:
        return map(int, s.split(","))
    else:
        return []

def discount(x, gamma):
    """
    computes discounted sums along 0th dimension of x.
    inputs
    ------
    x: ndarray
    gamma: float
    outputs
    -------
    y: ndarray with same shape as x, satisfying
        y[t] = x[t] + gamma*x[t+1] + gamma^2*x[t+2] + ... + gamma^k x[t+k],
                where k = len(x) - t - 1
    """
    assert x.ndim >= 1
    return scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]