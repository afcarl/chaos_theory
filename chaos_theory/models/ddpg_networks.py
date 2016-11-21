"""
Specialized networks for DDPG updates
"""
from collections import namedtuple
import tensorflow as tf
import numpy as np

from chaos_theory.models.tf_network import TFNet, TFContext
from chaos_theory.models.value import ValueDataPoint
from chaos_theory.utils.tf_utils import assert_shape
from chaos_theory.data import to_dataset

SARSData = namedtuple('DataPoint', ['obs', 'act', 'rew', 'obs_next'])

def compute_sars(traj):
    dataset = []
    for t in range(len(traj)-1):
        dataset.append(SARSData(traj.obs[t], traj.act[t], traj.rew[t], traj.obs[t+1]))
    return to_dataset(dataset)


class CriticQNetwork(TFContext):
    def __init__(self, sess, obs_space, action_space, network_arch, actor):
        self.dO = obs_space.shape[0]
        self.dU = action_space.shape[0]
        self.obs_space = obs_space
        self.action_space = action_space
        self.actor = actor
        self.build_network(network_arch, actor, self.dO, self.dU)
        super(CriticQNetwork, self).__init__(sess=sess)

    def build_network(self, network_arch, actor, dO, dU):
        with tf.variable_scope('critic_network') as vs:
            # Policy gradient
            self.lr = tf.placeholder(tf.float32, name='lr')
            self.obs = actor.obs_tensor
            self.action = actor.action_tensor
            assert_shape(self.obs, [None, dO])
            assert_shape(self.action, [None, dU])
            self.q_pol = network_arch(self.obs, self.action)
            self.actor_op = tf.train.AdamOptimizer(self.lr).minimize(-self.q_pol)

            # Supervised action
            self.sup_obs = tf.placeholder(tf.float32, [None, dO], name='sup_obs')
            self.sup_action = tf.placeholder(tf.float32, [None, dU], name='sup_act')
            self.q_labels = tf.placeholder(tf.float32, [None], name='sup_qlabels')
            self.q_pred = network_arch(self.sup_obs, self.sup_action, reuse=True)
            self.loss = tf.reduce_mean(tf.square(self.q_labels - self.q_pred))
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.minimize(self.loss)

            self.trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

    def update_policy(self, batch, lr):
        return self.run(self.actor_op, {self.lr: lr,
                                        self.obs: batch.stack.obs,
                                        self.action: batch.stack.act})

    @property
    def q_pol_tensor(self):
        return self.q_pol

    def train_step(self, batch, lr):
        return self.run([self.loss, self.train_op], {self.lr: lr,
                                                     self.sup_obs: batch.stack.obs,
                                                     self.sup_action: batch.stack.act,
                                                     self.q_labels: batch.stack.returns})[0]


class TargetQNetwork(TFContext):
    def __init__(self, sess, obs_space, action_space, network_arch, critic_network):
        self.dO = obs_space.shape[0]
        self.dU = action_space.shape[0]
        self.obs_space = obs_space
        self.action_space = action_space
        self.actor = critic_network.actor
        self.critic_network = critic_network
        self.build_network(network_arch)
        super(TargetQNetwork, self).__init__(sess=sess)

    def build_network(self, network_arch):
        with tf.variable_scope('target_network') as vs:
            self.obs = self.actor.obs_tensor
            self.action = self.actor.action_tensor
            self.q_pred = network_arch(self.obs, self.action)
            assert_shape(self.q_pred, [None, 1])

            self.track_rate = tf.placeholder(tf.float32, name='track_rate')
            self.trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

            self.track_ops = [tf.assign(self.trainable_vars[i],
                      self.track_rate*self.trainable_vars[i] + (1-self.track_rate)*self.critic_network.trainable_vars[i])
                              for i in range(len(self.trainable_vars))]

            self.copy_op = [tf.assign(self.trainable_vars[i], self.critic_network.trainable_vars[i])
                              for i in range(len(self.trainable_vars))]

    def compute_returns(self, batch):
        returns = self.run(self.q_pred, {self.obs: batch.stack.obs})
        returns = returns[:,0] + batch.stack.rew

        data = []
        for i in range(len(batch)):
            data.append(ValueDataPoint(batch[i].obs, batch[i].act, returns[i]))
        return to_dataset(data)


    def track(self, lr):
        self.sess.run(self.track_ops, {self.track_rate: lr})

    def copy_critic(self):
        self.sess.run(self.copy_op)

    def train_step(self, batch, lr):
        raise NotImplementedError()
