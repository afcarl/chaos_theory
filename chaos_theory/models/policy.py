import tensorflow as tf
import numpy as np

from chaos_theory.data import ListDataset
from chaos_theory.models.tf_network import TFNet
from chaos_theory.utils import linear, assert_shape


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

    def train_step(self, trajlist, lr):
        return self.network.train_step(ListDataset(trajlist), lr)


class PolicyNetwork(TFNet):
    def __init__(self, action_space, obs_space, policy_network=linear_gaussian_policy,
                 update_rule=ReinforceGrad):
        self.dO = obs_space.shape[0]
        self.dU = action_space.shape[0]
        super(PolicyNetwork, self).__init__(value_network=policy_network,
                                            update_rule=update_rule,
                                           dO=self.dO,
                                           dU=self.dU)
        self.obs_space = obs_space


    def build_network(self, policy_network, update_rule, dO, dU):
        self.obs = tf.placeholder(tf.float32, [None, dO])
        self.act = tf.placeholder(tf.float32, [None, dU])
        self.act_sample, self.act_prob = policy_network(self.obs, self.act, dU)

        self.lr = tf.placeholder(tf.float32)

        self.obs_surr= tf.placeholder(tf.float32, [None, dO])
        self.act_surr = tf.placeholder(tf.float32, [None, dU])
        self.returns_surr = tf.placeholder(tf.float32, [None, 1])
        self.surr_loss = update_rule.surr_loss(self.obs_surr, self.act_surr,
                                               self.returns_surr, policy_network)

        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.surr_loss)

    def sample_act(self, obs):
        obs = np.expand_dims(obs, axis=0)
        act = self.run([self.act_sample], {self.obs: obs})[0]
        return act[0]

    def train_step(self, batch, lr):
        return self.run([self.surr_loss, self.train_op], {self.lr: lr,
                                                     self.obs_surr: batch.concat.obs,
                                                     self.act_surr: batch.concat.act,
                                                     self.returns_surr: batch.concat.returns})[0]

def linear_gaussian_policy(obs, act, dU):
    with tf.variable_scope('policy') as vs:
        mu = linear(obs, dout=dU, name='mu')
        sigma = tf.exp(linear(obs, dout=dU, name='logsig'))
        #params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

    dist = tf.contrib.distributions.Normal(mu=mu[0], sigma=sigma[0])
    act_sample = dist.sample(1)
    act_prob = dist.pdf(act)
    return act_sample, act_prob


class ReinforceGrad():
    def __init__(self, advantage=lambda x: x):
        self.advantage_fn = advantage

    def surr_loss(self, obs_tensor, act_tensor, returns_tensor, policy):
        # Compute advantages
        advantage = self.advantage_fn(returns_tensor)
        assert_shape(advantage, returns_tensor.get_shape())

        # Compute surr loss
        prob_act = policy.act_prob_tensor(obs_tensor, act_tensor)
        log_prob_act = tf.log(prob_act)
        assert_shape(log_prob_act, [None, 1])

        surr_loss = tf.reduce_sum(log_prob_act * advantage)
        return surr_loss
