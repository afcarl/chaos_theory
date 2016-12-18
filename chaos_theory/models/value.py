"""
Value function networks
"""
from collections import namedtuple

import tensorflow as tf

from .tf_network import TFNet


ValueDataPoint = namedtuple('DataPoint', ['obs', 'act', 'returns'])


def create_value_datapoints(traj):
    dataset = []
    for t in range(len(traj)):
        dataset.append(ValueDataPoint(traj.obs[t], traj.act[t], traj.returns[t]))
    return dataset


class ValueNetwork(TFNet):
    def __init__(self, tf_context, obs_space, value_network_def,
                 obs_tensor=None, labels_tensor=None):
        super(ValueNetwork, self).__init__()
        self.context = tf_context
        self.dO = obs_space.shape[0]
        self.obs_space = obs_space
        if obs_tensor is None:
            self.obs = tf.placeholder(tf.float32, [None, self.dO])
        else:
            self.obs = obs_tensor

        if labels_tensor is None:
            self.value_labels = tf.placeholder(tf.float32, [None])
        else:
            self.value_labels = labels_tensor

        self.lr = tf.placeholder(tf.float32)
        with tf.variable_scope('value_network'):
            self.value_pred = value_network_def(self.obs)
        self.loss = tf.reduce_mean(tf.square(self.value_labels - self.value_pred))
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.loss)

    def train_step(self, batch, lr):
        return self.context.run([self.loss, self.train_op], {self.lr:lr,
                                                     self.obs: batch.stack.obs,
                                                     self.value_labels: batch.stack.returns})[0]


class QNetwork(TFNet):
    def __init__(self, tf_context, obs_space, action_space, q_network_def,
                 obs_tensor=None, action_tensor=None, labels_tensor=None):
        super(QNetwork, self).__init__()
        self.context = tf_context
        self.dO = obs_space.shape[0]
        self.dU = action_space.shape[0]
        self.obs_space = obs_space
        self.action_space = action_space

        self.obs = obs_tensor
        self.action = action_tensor
        self.q_labels = labels_tensor

        if obs_tensor is None:
            self.obs = tf.placeholder(tf.float32, [None, self.dO])
        if action_tensor is None:
            self.action = tf.placeholder(tf.float32, [None, self.dU])
        if labels_tensor is None:
            self.q_labels = tf.placeholder(tf.float32, [None])
        self.lr = tf.placeholder(tf.float32)
        self.q_pred = q_network_def(self.obs, self.action)
        self.loss = tf.reduce_mean(tf.square(self.q_labels - self.q_pred))
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.loss)

    def train_step(self, batch, lr):
        return self.context.run([self.loss, self.train_op], {self.lr: lr,
                                                     self.obs: batch.stack.obs,
                                                     self.action: batch.stack.act,
                                                     self.q_labels: batch.stack.returns})[0]

