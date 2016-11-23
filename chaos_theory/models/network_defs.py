import tensorflow as tf
from gym.spaces import Box

from chaos_theory.distribution import DiagGauss
from chaos_theory.distribution.categorical import SoftmaxDistribution
from chaos_theory.utils import linear, assert_shape


def two_layer_policy(l1=200, l2=200, action_scale=1.0):
    def policy(obs, dimA, reuse=False):
        with tf.variable_scope('policy', reuse=reuse):
            h1 = tf.nn.relu(linear(obs, dout=l1, name='h1'))
            h2 = tf.nn.relu(linear(h1, dout=l2, name='h2'))
            h3 = tf.identity(linear(h2, dout=dimA), name='h3')
            action = tf.nn.tanh(h3, name='h4-action') * action_scale
            return action
    return policy


def two_layer_q(l1=200, l2=200):
    def qfunction(obs, act, reuse=False):
        with tf.variable_scope('qfunc', reuse=reuse):
            h1 = tf.nn.relu(linear(obs, dout=l1, name='h1'))
            h1a = tf.concat(1, [h1, act])
            h2 = tf.nn.relu(linear(h1a, dout=l2, name='h2'))
            qs = linear(h2, dout=1)
            q = tf.squeeze(qs, [1], name='h3-q')
            return q
    return qfunction


def linear_softmax_policy():
    def inner(obs, dU, reuse=False):
        dist = SoftmaxDistribution(dU)
        with tf.variable_scope('policy', reuse=reuse) as vs:
            dist.compute_params_tensor(obs)
        return dist
    return inner


def linear_gaussian_policy(min_std=0.1):
    def inner(obs, dU, reuse=False):
        dist = DiagGauss(dU, min_var=min_std)
        with tf.variable_scope('policy', reuse=reuse) as vs:
            dist.compute_params_tensor(obs)
        return dist
    return inner


def tanh_deterministic_policy(action_space, dim_hidden=10, num_hidden=0):
    assert isinstance(action_space, Box)
    low = action_space.low
    high = action_space.high
    mid = (low+high)/2
    diff = high-low
    def inner(obs, dU, reuse=False):
        out = obs
        with tf.variable_scope('policy', reuse=reuse):
            for i in range(num_hidden):
                out = tf.nn.relu(linear(out, dout=dim_hidden, name='layer_%d'%i))
            out = linear(out, dout=dU, init_scale=0.01)
            pol = tf.nn.tanh(out)*(diff/2) + mid
        return pol
    return inner


def relu_gaussian_policy(num_hidden=1, dim_hidden=10, min_std=0.0, mean_clamp=None):
    def inner(obs, dU, reuse=False):
        out = obs
        dist = DiagGauss(dU, mean_clamp=mean_clamp, min_var=min_std)
        with tf.variable_scope('policy', reuse=reuse) as vs:
            for i in range(num_hidden):
                out = tf.nn.relu(linear(out, dout=dim_hidden, name='layer_%d'%i))
            dist.compute_params_tensor(out)
        return dist
    return inner


def relu_q_fn(num_hidden=1, dim_hidden=10):
    def inner(state, action, reuse=False):
        sout = state
        dU = int(action.get_shape()[1])
        with tf.variable_scope('q_function', reuse=reuse) as vs:
            for i in range(num_hidden):
                sout = tf.nn.relu(linear(sout, dout=dim_hidden, name='layer_%d'%i))
            sa = tf.concat(1, [sout, action])
            assert_shape(sa, [None, dim_hidden+dU])
            out = tf.nn.relu(linear(sa, dout=dim_hidden, name='sa1'))
            out = linear(out, dout=1, init_scale=0.01, name='sa2')
        return out
    return inner


def linear_q_fn(state, action, reuse=False):
    with tf.variable_scope('q_function', reuse=reuse):
        a1 = linear(state, dout=1, name='state')
        a2 = linear(action, dout=1, name='act')
    return a1+a2


def linear_value_fn(state):
    value = linear(state, dout=1)
    return value