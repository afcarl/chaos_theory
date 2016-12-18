import tensorflow as tf
import numpy as np
import logging

from chaos_theory.data import FIFOBuffer, ListDataset
from chaos_theory.models.value import ValueNetwork, create_value_datapoints
from chaos_theory.models.network_defs import linear_value_fn
from chaos_theory.utils import assert_shape

LOGGER = logging.getLogger(__name__)


class Advantage(object):
    def __init__(self):
        pass

    def apply(self, returns, actions, obs):
        return returns

    def update(self, samples):
        pass


class LinearBaseline(Advantage):
    def __init__(self, context, obs_space):
        super(LinearBaseline, self).__init__()
        self.context = context
        self.obs_space = obs_space

    def eval(self, obs):
        obs = np.expand_dims(obs, axis=0)
        return self.context.run(self.network.value_pred, {self.network.obs: obs})[0]

    def apply(self, ret, act, obs):
        self.network = ValueNetwork(self.context, self.obs_space, linear_value_fn,
                     obs_tensor=obs, labels_tensor=ret)
        assert_shape(self.network.value_pred, [None, 1])
        value_pred = self.network.value_pred[:,0]
        adv = ret - value_pred
        return adv

    def update(self, samples):
        training_data = ListDataset(samples)
        self.network.fit(training_data, max_iter=2000, heartbeat=500)

    def train_step(self, batch, lr):
        self.network.train_step(batch, lr)

class GAE(Advantage):
    def __init__(self, obs_space, _lambda=0.97):
        super(GAE, self).__init__()
        self.obs_space = obs_space
        raise NotImplementedError()
