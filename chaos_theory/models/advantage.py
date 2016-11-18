import tensorflow as tf
import numpy as np
import logging

from chaos_theory.data import FIFOBuffer, ListDataset
from chaos_theory.models.value import ValueNetwork, linear_value_fn, create_value_datapoints
from chaos_theory.utils import assert_shape

LOGGER = logging.getLogger(__name__)


class Advantage(object):
    def __init__(self):
        pass

    def apply(self, returns, actions, obs):
        return returns

    def update(self, samples):
        pass


class LinearBaseline(ValueNetwork, Advantage):
    def __init__(self, obs_space):
        super(LinearBaseline, self).__init__(obs_space, linear_value_fn)

    def eval(self, obs):
        obs = np.expand_dims(obs, axis=0)
        return self.run(self.value_pred, {self.obs: obs})[0]

    def apply(self, ret, act, obs):
        with tf.variable_scope('value_network', reuse=True):
            value_pred = self.build_value_network(obs)
        assert_shape(value_pred, [None, 1])
        value_pred = value_pred[:,0]
        adv = ret - value_pred
        return adv

    def update(self, samples):
        training_data = ListDataset(samples)
        self.fit(training_data, max_iter=2000, heartbeat=500)

    def train_step(self, batch, lr):
        return self.run([self.loss, self.train_op], {self.lr:lr,
                                                     self.obs: batch.concat.obs,
                                                     self.value_labels: batch.concat.returns})[0]


class GAE(Advantage):
    def __init__(self, obs_space, _lambda=0.97):
        super(GAE, self).__init__()
        self.obs_space = obs_space
        raise NotImplementedError()
