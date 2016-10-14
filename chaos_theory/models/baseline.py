import tensorflow as tf
import numpy as np
import logging
from collections import namedtuple

from chaos_theory.utils import discount_value, BatchSampler
from chaos_theory.utils.tf_utils import linear

LOGGER = logging.getLogger(__name__)

DataPoint = namedtuple('DataPoint', ['obs', 'value'])

class LinearBaseline(object):
    def __init__(self, dim_obs):
        self.graph = tf.Graph()
        self.dim_obs = dim_obs
        with self.graph.as_default():
            self._build_model()
            self._init_tf()
        self.train_dataset = []

    def add_to_buffer(self, traj, discount=0.99):
        obs = traj.obs
        act = traj.rew
        values = discount_value(act)
        for t in range(len(values)):
            self.train_dataset.append(DataPoint(obs[t], values[t]))

    def clear_buffer(self):
        self.train_dataset = []

    def _build_model(self):
        self.obs = tf.placeholder(tf.float32, [None, self.dim_obs])
        self.value_labels = tf.placeholder(tf.float32, [None, 1])
        self.lr = tf.placeholder(tf.float32)

        with tf.variable_scope('baseline'):
            l1 = tf.nn.relu(linear(self.obs, dout=10, name='l1'))
            self.value = linear(l1, dout=1, name='obs')

        self.loss = tf.reduce_mean(tf.square(self.value_labels-self.value))
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.loss)

    def _init_tf(self):
        self.sess = tf.Session()
        self.run(tf.initialize_all_variables())

    def _make_feed(self, batch):
        obs_batch = np.r_[[datum.obs for datum in batch]]
        val_batch = np.r_[[datum.value for datum in batch]]
        val_batch = np.expand_dims(val_batch, axis=-1)
        return {self.obs: obs_batch, self.value_labels:val_batch}

    def run(self, fetches, feeds=None):
        with self.graph.as_default():
            return self.sess.run(fetches, feed_dict=feeds)

    def train_step(self, batch, lr=1e-3):
        feeds = self._make_feed(batch)
        feeds[self.lr] = lr
        loss, _ = self.run([self.loss, self.train_op], feeds=feeds)
        return loss

    def eval(self, obs):
        obs = np.expand_dims(obs, 0)
        val = self.run(self.value, feeds={self.obs: obs})
        return float(val[0])

    def train(self, batch_size=5, heartbeat=500, max_iter=5000):
        sampler = BatchSampler(self.train_dataset)
        avg_loss = 0
        for i, batch in enumerate(sampler.with_replacement(batch_size=batch_size)):
            loss = self.train_step(batch)
            avg_loss += loss
            if i%heartbeat == 0 and i>0:
                LOGGER.debug('Itr %d: Loss %f', i, avg_loss/heartbeat)
                avg_loss=0
            if i > max_iter:
                break

        
