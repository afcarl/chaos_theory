"""
Second DDPG implementation, based on ICNN code (https://github.com/locuslab/icnn)

Slightly faster, due to returns being calculated within the TF graph.
"""
import numpy as np
import tensorflow as tf

from chaos_theory.algorithm.algorithm import OnlineAlgorithm
from chaos_theory.data import FIFOBuffer, BatchSampler
from chaos_theory.models.ddpg_networks import SARSData
from chaos_theory.models.exploration import OUStrategy
from chaos_theory.models.network_defs import two_layer_policy, two_layer_q
from chaos_theory.models.policy import Policy
from chaos_theory.utils import make_track_op


class DDPG(OnlineAlgorithm, Policy):
    def __init__(self, env,
                 pol_network=two_layer_policy(),
                 q_network=two_layer_q(),
                 track_tau=0.001,
                 discount=0.95,
                 actor_lr=1e-4,
                 q_lr=1e-3,
                 noise_sigma=0.2,
                 noise_theta=0.15,
                 rbuffer_size=1e5,
                 min_buffer_size=1e3,
                 l2_reg=0,
                 inner_itrs=1,
                 batch_size=32
                 ):
        super(DDPG, self).__init__()
        dO = env.observation_space.shape[0]
        dU = env.action_space.shape[0]

        self.min_buffer_size = min_buffer_size
        self.inner_itrs = inner_itrs
        self.batch_size = batch_size
        self.buffer = FIFOBuffer(rbuffer_size)
        self.sess = tf.Session()

        self.exploration = OUStrategy(env.action_space, theta=noise_theta, sigma=noise_sigma)
        self.obs = tf.placeholder(tf.float32, [None, dO], "obs")
        self.act_train = tf.placeholder(tf.float32, [None, dU], "act_train")
        self.rew = tf.placeholder(tf.float32, [None], "rew")
        self.obs_next = tf.placeholder(tf.float32, [None, dO], "obs_next")
        self.done = tf.placeholder(tf.bool, [None], "term2")

        # Policy loss
        with tf.variable_scope('pol') as vs:
            self.act_test = pol_network(self.obs, dU)
            vs.reuse_variables()
            act_train_policy = pol_network(self.obs, dU)
        with tf.variable_scope('critic_q'):
            q_train_policy = q_network(self.obs, act_train_policy)
        meanq = tf.reduce_mean(q_train_policy, reduction_indices=0)
        pol_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pol')
        wd_p = tf.add_n([l2_reg * tf.nn.l2_loss(var) for var in pol_vars])  # weight decay
        loss_p = -meanq + wd_p
        self.pol_optimize_op = tf.train.AdamOptimizer(learning_rate=actor_lr).minimize(loss_p, var_list=pol_vars)

        # Q loss
        with tf.variable_scope('critic_q', reuse=True):
            q_train = q_network(self.obs, self.act_train)
        with tf.variable_scope('target_pol'):
            act2 = pol_network(self.obs_next, dU)
        with tf.variable_scope('target_q'):
            q2 = q_network(self.obs_next, act2)
        q_target = tf.stop_gradient(tf.select(self.done, self.rew, self.rew + discount * q2))
        td_error = q_train - q_target
        ms_td_error = tf.reduce_mean(tf.square(td_error), reduction_indices=0)
        q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_q')
        wd_q = tf.add_n([l2_reg * tf.nn.l2_loss(var) for var in q_vars])  # weight decay
        loss_q = ms_td_error + wd_q
        self.q_optimize_op = tf.train.AdamOptimizer(learning_rate=q_lr).minimize(loss_q, var_list=q_vars)

        # Tracking OPs
        self.pol_track_op = make_track_op('target_pol', 'pol', track_tau)
        self.q_track_op= make_track_op('target_q', 'critic_q', track_tau)

        self.sess.run(tf.initialize_all_variables())

    def update(self, s, a, r, sn, done):
        self.buffer.append(SARSData(s,a,r, sn, 0 if done else 1))

        if len(self.buffer) < self.min_buffer_size:
            return

        for batch in BatchSampler(self.buffer).with_replacement(batch_size=self.batch_size,
                                                                max_itr=self.inner_itrs):
            obs, act, rew, ob2, not_done = (batch.stack.obs, batch.stack.act, batch.stack.rew,
                                         batch.stack.obs_next, batch.stack.term_mask)
            done = (1.0-not_done).astype(np.bool)

            # Optimize policy
            self.sess.run(self.pol_optimize_op, {self.obs: obs})
            self.sess.run(self.q_optimize_op, {self.obs: obs, self.act_train: act, self.rew: rew,
                                               self.obs_next: ob2, self.done: done})

            # Target tracking
            self.sess.run(self.q_track_op)
            self.sess.run(self.pol_track_op)

    def reset(self):
        self.exploration.reset()

    def act(self, obs):
        obs = np.expand_dims(obs, axis=0)
        act = self.sess.run(self.act_test, feed_dict={self.obs: obs})
        action = self.exploration.add_noise(act)
        self.action = np.atleast_1d(np.squeeze(action, axis=0))
        return self.action

    def get_policy(self):
        return self

    def __del__(self):
        self.sess.close()



