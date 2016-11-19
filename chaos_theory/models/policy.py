import tensorflow as tf
import numpy as np
import logging

from chaos_theory.algorithm.reinforce import ReinforceGrad
from chaos_theory.data import ListDataset
from chaos_theory.models.tf_network import TFNet
from chaos_theory.utils import linear, assert_shape
from chaos_theory.distribution import DiagGauss

LOGGER = logging.getLogger(__name__)


class Policy(object):
    """docstring for Policy"""
    def __init__(self):
        super(Policy, self).__init__()

    def act(self, obs):
        raise NotImplementedError()


class RandomPolicy(Policy):
    """docstring for RandomPolicy"""
    def __init__(self, action_space):
        super(RandomPolicy, self).__init__()
        self.action_space = action_space

    def act(self, obs):
        return self.action_space.sample()


class ContinuousPolicy(Policy):
    def __init__(self, network):
        super(ContinuousPolicy, self).__init__()
        self.network = network

    def act(self, obs):
        return self.network.sample_act(obs)

    def act_entropy(self, obs):
        return self.network.action_entropy(obs)

    def train_step(self, trajlist, lr):
        return self.network.train_step(ListDataset(trajlist), lr)


def linear_gaussian_policy(min_std=0.1):
    def inner(obs, dU, reuse=False):
        dist = DiagGauss(dU, min_var=min_std)
        with tf.variable_scope('policy', reuse=reuse) as vs:
            dist.compute_params_tensor(obs)
        return dist
    return inner


def relu_policy(num_hidden=1, dim_hidden=10, min_std=0.0, mean_clamp=None):
    def inner(obs, dU, reuse=False):
        out = obs
        dist = DiagGauss(dU, mean_clamp=mean_clamp, min_var=min_std)
        with tf.variable_scope('policy', reuse=reuse) as vs:
            for i in range(num_hidden):
                out = tf.nn.relu(linear(out, dout=dim_hidden, name='layer_%d'%i))
            dist.compute_params_tensor(out)
        return dist
    return inner


class PolicyNetwork(TFNet):
    def __init__(self, action_space, obs_space, 
                policy_network=linear_gaussian_policy(0.01),
                ):
        self.dO = obs_space.shape[0]
        self.dU = action_space.shape[0]
        super(PolicyNetwork, self).__init__(policy_network=policy_network,
                                           dO=self.dO,
                                           dU=self.dU)
        self.obs_space = obs_space
        self.action_space = action_space
        self.policy_network = policy_network

    def build_network(self, policy_network, dO, dU):
        self.obs = tf.placeholder(tf.float32, [None, dO], name='obs')
        self.act = tf.placeholder(tf.float32, [None, dU], name='act')
        self.pol_dist = policy_network(self.obs, dU)

    def __sample_act(self, obs):
        obs = np.expand_dims(obs, axis=0)
        pol_params = self.run(self.pol_dist.params, {self.obs: obs})
        return self.pol_dist.sample(1, *pol_params)[0]

    def sample_act(self, obs):
        act = self.__sample_act(obs)
        return act

    def action_entropy(self, obs):
        if len(obs.shape) < 2:
            obs = np.expand_dims(obs, 0)
        pol_params = self.run(self.pol_dist.params, {self.obs:obs})
        return self.pol_dist.entropy(*pol_params)

    def train_step(self, batch, lr):
        raise NotImplementedError()
