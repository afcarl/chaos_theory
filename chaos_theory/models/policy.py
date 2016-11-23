import tensorflow as tf
import numpy as np
import logging

from gym.spaces import Box

from chaos_theory.algorithm.reinforce import ReinforceGrad
from chaos_theory.data import ListDataset
from chaos_theory.distribution.categorical import SoftmaxDistribution
from chaos_theory.models.tf_network import TFNet
from chaos_theory.utils import linear, assert_shape
from chaos_theory.distribution import DiagGauss
from chaos_theory.utils.gym_utils import action_space_dim

LOGGER = logging.getLogger(__name__)


def linear_gaussian_policy(min_std=0.1):
    def inner(obs, dU, reuse=False):
        dist = DiagGauss(dU, min_var=min_std)
        with tf.variable_scope('policy', reuse=reuse) as vs:
            dist.compute_params_tensor(obs)
        return dist
    return inner


def linear_softmax_policy():
    def inner(obs, dU, reuse=False):
        dist = SoftmaxDistribution(dU)
        with tf.variable_scope('policy', reuse=reuse) as vs:
            dist.compute_params_tensor(obs)
        return dist
    return inner


def tanh_deterministic_policy(action_space, act_slack=1.0, dim_hidden=10, num_hidden=0):
    assert isinstance(action_space, Box)
    low = action_space.low
    high = action_space.high
    mid = (low+high)/2
    diff = high-low
    diff *= act_slack
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


class GreedyPolicy(Policy):
    def __init__(self, q_net):
        super(GreedyPolicy, self).__init__()
        self.q_net = q_net

    def act(self, obs):
        raise NotImplementedError()


class NNPolicy(Policy):
    def __init__(self, network):
        super(NNPolicy, self).__init__()
        self.network = network

    def act(self, obs):
        return self.network.sample_act(obs)

    def act_entropy(self, obs):
        return self.network.action_entropy(obs)

    def reset(self):
        self.network.reset()


class StochasticPolicyNetwork(TFNet):
    def __init__(self, action_space, obs_space, 
                policy_network=linear_gaussian_policy(0.01),
                ):
        self.dO = obs_space.shape[0]
        self.dU = action_space_dim(action_space)
        super(StochasticPolicyNetwork, self).__init__(policy_network=policy_network,
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
        return self.pol_dist.sample(1, pol_params)[0]

    def sample_act(self, obs):
        act = self.__sample_act(obs)
        return act

    def action_entropy(self, obs):
        if len(obs.shape) < 2:
            obs = np.expand_dims(obs, 0)
        pol_params = self.run(self.pol_dist.params, {self.obs:obs})
        return self.pol_dist.entropy(pol_params)

    def train_step(self, batch, lr):
        raise NotImplementedError()


class DeterministicPolicyNetwork(TFNet):
    def __init__(self, action_space, obs_space,
                 policy_network,
                 exploration=None,
                 ):
        self.dO = obs_space.shape[0]
        self.dU = action_space_dim(action_space)
        super(DeterministicPolicyNetwork, self).__init__(policy_network=policy_network,
                                                         dO=self.dO,
                                                         dU=self.dU)
        self.obs_space = obs_space
        self.action_space = action_space
        self.policy_network = policy_network
        self.exploration=exploration

    def build_network(self, policy_network, dO, dU):
        with tf.variable_scope('policy') as vs:
            self.obs = tf.placeholder(tf.float32, [None, dO], name='obs')
            self.pol_out = policy_network(self.obs, dU)
            self.trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

    def sample_act(self, obs):
        obs = np.expand_dims(obs, axis=0)
        act = self.run(self.pol_out, {self.obs: obs})[0]
        if self.exploration:
            act = self.exploration.add_noise(act)
        return act

    def reset(self):
        self.exploration.reset()

    def action_entropy(self, obs):
        return 0

    def train_step(self, batch, lr):
        raise NotImplementedError()

    @property
    def action_tensor(self):
        return self.pol_out

    @property
    def obs_tensor(self):
        return self.obs


