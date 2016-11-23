"""
Value function networks
"""
from collections import namedtuple

import tensorflow as tf

from chaos_theory.utils import assert_shape
from chaos_theory.utils import linear
from .tf_network import TFNet


ValueDataPoint = namedtuple('DataPoint', ['obs', 'act', 'returns'])


def create_value_datapoints(traj):
    dataset = []
    for t in range(len(traj)):
        dataset.append(ValueDataPoint(traj.obs[t], traj.act[t], traj.returns[t]))
    return dataset


class ValueNetwork(TFNet):
    def __init__(self, obs_space, value_network):
        self.dO = obs_space.shape[0]
        super(ValueNetwork, self).__init__(value_network=value_network,
                                           dO=self.dO)
        self.build_value_network = value_network
        self.obs_space = obs_space

    def build_network(self, value_network, dO):
        self.obs = tf.placeholder(tf.float32, [None, dO])
        self.value_labels = tf.placeholder(tf.float32, [None])
        self.lr = tf.placeholder(tf.float32)
        with tf.variable_scope('value_network'):
            self.value_pred = value_network(self.obs)
        self.loss = tf.reduce_mean(tf.square(self.value_labels - self.value_pred))
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.loss)

    def train_step(self, batch, lr):
        return self.run([self.loss, self.train_op], {self.lr:lr,
                                                     self.obs: batch.stack.obs,
                                                     self.value_labels: batch.stack.returns})[0]


class QNetwork(TFNet):
    def __init__(self, obs_space, action_space, q_network):
        self.dO = obs_space.shape[0]
        self.dU = action_space.shape[0]
        super(QNetwork, self).__init__(q_network=q_network,
                                       dO=self.dO,
                                       dU=self.dU)
        self.obs_space = obs_space
        self.action_space = action_space


    def build_network(self, q_network, dO, dU):
        self.obs = tf.placeholder(tf.float32, [None, dO])
        self.action = tf.placeholder(tf.float32, [None, dU])
        self.q_labels = tf.placeholder(tf.float32, [None])
        self.lr = tf.placeholder(tf.float32)
        self.q_pred = q_network(self.obs, self.action)
        self.loss = tf.reduce_mean(tf.square(self.q_labels - self.q_pred))
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.loss)

    def train_step(self, batch, lr):
        return self.run([self.loss, self.train_op], {self.lr: lr,
                                                     self.obs: batch.stack.obs,
                                                     self.action: batch.stack.act,
                                                     self.q_labels: batch.stack.returns})[0]

def linear_value_fn(state):
    value = linear(state, dout=1)
    return value


def linear_q_fn(state, action, reuse=False):
    with tf.variable_scope('q_function', reuse=reuse):
        a1 = linear(state, dout=1, name='state')
        a2 = linear(action, dout=1, name='act')
    return a1+a2


def pointmass_q_star(state, action, reuse=False):
    with tf.variable_scope('q_function', reuse=reuse):
        a1 = linear(state, dout=1, name='state')*0.01
        a2 = linear(action, dout=1, bias=False, name='act') + 0.2*action
    return a1+a2


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
