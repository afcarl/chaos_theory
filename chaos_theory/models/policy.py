import tensorflow as tf
import numpy as np
import logging

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


class ReinforceGrad():
    def __init__(self, advantage=lambda x: x):
        self.advantage_fn = advantage

    def surr_loss(self, obs_tensor, act_tensor, returns_tensor, batch_size, pol_network):
        # Compute advantages
        advantage = self.advantage_fn(returns_tensor)
        #avg_advantage = tf.reduce_mean(advantage)
        #advantage = advantage/avg_advantage
        advantage = tf.Print(advantage, [advantage], summarize=10, message='Advtg')
        assert_shape(advantage, [None])

        # Compute surr loss
        dU = int(act_tensor.get_shape()[-1])
        sur_pol_dist = pol_network(obs_tensor, dU, reuse=True)
        act_tensor = tf.Print(act_tensor, [act_tensor], summarize=10, message='actions')
        log_prob_act = sur_pol_dist.log_prob_tensor(act_tensor)
        assert_shape(log_prob_act, [None])

        # debugging
        prob_act = tf.exp(log_prob_act)
        prob_act = tf.Print(prob_act, [prob_act], summarize=10, message='like')
        log_prob_act = tf.log(prob_act)

        #log_prob_act = tf.Print(log_prob_act, [log_prob_act], summarize=10, message='Logli')


        weighted_logprob = log_prob_act * advantage
        #weighted_logprob = tf.Print(weighted_logprob, [weighted_logprob], summarize=10, message='WeightLogli')
        surr_loss = - tf.reduce_sum(weighted_logprob)/batch_size
        return surr_loss


class PolicyNetwork(TFNet):
    def __init__(self, action_space, obs_space, 
                policy_network=linear_gaussian_policy(0.01),
                 update_rule=ReinforceGrad()):
        self.dO = obs_space.shape[0]
        self.dU = action_space.shape[0]
        super(PolicyNetwork, self).__init__(policy_network=policy_network,
                                            update_rule=update_rule,
                                           dO=self.dO,
                                           dU=self.dU)
        self.obs_space = obs_space
        self.action_space = action_space


    def build_network(self, policy_network, update_rule, dO, dU):
        self.obs = tf.placeholder(tf.float32, [None, dO], name='obs')
        self.act = tf.placeholder(tf.float32, [None, dU], name='act')
        self.pol_dist = policy_network(self.obs, dU)

        self.lr = tf.placeholder(tf.float32)

        self.obs_surr= tf.placeholder(tf.float32, [None, dO], name='obs_surr')
        self.act_surr = tf.placeholder(tf.float32, [None, dU], name='act_surr')
        self.returns_surr = tf.placeholder(tf.float32, [None], name='ret_surr')
        self.batch_size = tf.placeholder(tf.float32, (), name='batch_size')
        self.surr_loss = update_rule.surr_loss(self.obs_surr, self.act_surr,
                                               self.returns_surr, self.batch_size,
                                               policy_network)

        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.surr_loss)
        self.gradients = [tf.gradients(self.surr_loss, param)[0] for param in tf.trainable_variables()]

    def __sample_act(self, obs):
        obs = np.expand_dims(obs, axis=0)
        pol_params = self.run(self.pol_dist.params, {self.obs: obs})
        return self.pol_dist.sample(1, *pol_params)[0]

    def sample_act(self, obs):
        act = self.__sample_act(obs)
        while not self.action_space.contains(act):
            #LOGGER.debug('Action not in action space: %s', act)
            act = self.__sample_act(obs)
        return act

    def train_step(self, batch, lr):
        return self.run([self.surr_loss, self.train_op], {self.lr: lr,
                                                     self.obs_surr: batch.concat.obs,
                                                     self.act_surr: batch.concat.act,
                                                     self.batch_size: len(batch),
                                                     self.returns_surr: batch.concat.returns})[0]


