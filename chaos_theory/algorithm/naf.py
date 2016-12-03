"""
NAF implementation, based on ICNN code (https://github.com/locuslab/icnn)
"""
import numpy as np
import tensorflow as tf

from chaos_theory.algorithm.algorithm import OnlineAlgorithm
from chaos_theory.data import BatchSampler, FIFOBuffer
from chaos_theory.models.ddpg_networks import SARSData
from chaos_theory.models.exploration import OUStrategy
from chaos_theory.models.policy import Policy


class NAF(OnlineAlgorithm, Policy):
    def __init__(self, env, tau=0.001, discount=0.95,
                 outheta=0.15, ousigma=0.2, lr=1e-3, l2reg=1e-3,
                 min_replay_size=1e3, rbuffer_size=1e5, batch_size=32):
        super(NAF, self).__init__()
        dimO = env.observation_space.shape
        dimA = env.observation_space.shape
        dimA = list(dimA)
        dimO = list(dimO)

        l1size = 200
        l2size = 200

        self.batch_size = batch_size
        self.min_buffer_size = min_replay_size
        self.exploration = OUStrategy(env.action_space, theta=outheta, sigma=ousigma)
        self.buffer = FIFOBuffer(rbuffer_size)
        self.sess = tf.Session()

        self.is_training = is_training = tf.placeholder(tf.bool)

        # create tf computational graph
        self.theta_L = make_params(dimO[0], dimA[0] * dimA[0], l1size, l2size, 'theta_L')
        self.theta_U = make_params(dimO[0], dimA[0], l1size, l2size, 'theta_U')
        self.theta_V = make_params(dimO[0], 1, l1size, l2size, 'theta_V')
        self.theta_Vt, update_Vt = exponential_moving_averages(self.theta_V, tau)

        self.obs_test = tf.placeholder(tf.float32, [1] + dimO, "obs-single")
        self.act_test = ufunction(self.obs_test, self.theta_U, False, is_training)

        # training
        self.obs = obs_train = tf.placeholder(tf.float32, [None] + dimO, 'obs_train')
        self.act_tensor = act_train = tf.placeholder(tf.float32, [None] + dimA, "act_train")
        self.rew = rew = tf.placeholder(tf.float32, [None], "rew")
        self.obs_next = obs2 = tf.placeholder(tf.float32, [None] + dimO, "obs2")
        self.done = term2 = tf.placeholder(tf.bool, [None], "term2")
        # q
        lmat = lfunction(obs_train, self.theta_L, False, is_training)
        uvalue = ufunction(obs_train, self.theta_U, True, is_training)
        avalue = afunction(act_train, lmat, uvalue, dimA[0])
        q_train = qfunction(obs_train, avalue, self.theta_V, False, is_training)

        # q targets
        q2 = qfunction(obs2, tf.constant([0.] * self.batch_size),
                            self.theta_Vt, True, is_training)
        q_target = tf.stop_gradient(tf.select(term2, rew, rew + discount * q2))

        # q loss
        td_error = q_train - q_target
        ms_td_error = tf.reduce_mean(tf.square(td_error), 0)
        theta = self.theta_L + self.theta_U + self.theta_V
        wd_q = tf.add_n([l2reg * tf.nn.l2_loss(var) for var in theta])  # weight decay
        loss_q = ms_td_error + wd_q
        # q optimization
        self.q_optimize_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_q)
        self.q_track_op = update_Vt

        # initialize tf variables
        self.sess.run(tf.initialize_all_variables())
        self.sess.graph.finalize()

    def get_policy(self):
        return self

    def reset(self):
        self.exploration.reset()

    def act(self, obs):
        obs = np.expand_dims(obs, axis=0)
        act = self.sess.run(self.act_test, feed_dict={self.obs_test: obs, self.is_training: False})
        action = self.exploration.add_noise(act)
        self.action = np.atleast_1d(np.squeeze(action, axis=0))
        return self.action

    def update(self, s, a, r, sn, done):
        self.buffer.append(SARSData(s,a,r, sn, 0 if done else 1))
        if len(self.buffer) < self.min_buffer_size:
            return

        for batch in BatchSampler(self.buffer).with_replacement(batch_size=self.batch_size,
                                                                max_itr=1):
            obs, act, rew, ob2, not_done = (batch.stack.obs, batch.stack.act, batch.stack.rew,
                                            batch.stack.obs_next, batch.stack.term_mask)
            done = (1.0-not_done).astype(np.bool)

            # Optimize policy
            feed_dict = {self.obs: obs, self.act_tensor: act, self.rew: rew, self.obs_next: ob2, self.done: done,
                         self.is_training: True}
            self.sess.run(self.q_optimize_op, feed_dict)

            # Target tracking
            self.sess.run(self.q_track_op)

    def __del__(self):
        self.sess.close()


def exponential_moving_averages(theta, tau=0.001):
    ema = tf.train.ExponentialMovingAverage(decay=1 - tau)
    update = ema.apply(theta)  # also creates shadow vars
    averages = [ema.average(x) for x in theta]
    return averages, update


def make_params(dimIn, dimOut, l1, l2, scope, initstd=0.01):
    with tf.variable_scope(scope):
        normal_init = tf.truncated_normal_initializer(mean=0.0, stddev=initstd)
        return [tf.get_variable(name='w1', shape=[dimIn, l1], initializer=normal_init),
                tf.get_variable(name='b1', shape=[l1], initializer=tf.constant_initializer(0.0)),
                tf.get_variable(name='w2', shape=[l1, l2], initializer=normal_init),
                tf.get_variable(name='b2', shape=[l2], initializer=tf.constant_initializer(0.0)),
                tf.get_variable(name='w3', shape=[l2, dimOut], initializer=normal_init),
                tf.get_variable(name='b3', shape=[dimOut], initializer=tf.constant_initializer(0.0))]


def build_NN_two_hidden_layers(x, theta, reuse, is_training, batch_norm=True):
    batch_norm_params = {'is_training': is_training, 'decay': 0.999, 'epsilon': 1e-3,
                         'updates_collections': None, 'reuse': reuse}
    if batch_norm:
        h0 = tf.contrib.layers.batch_norm(x, scope='h0', **batch_norm_params)
    else:
        h0 = x
    h1 = tf.matmul(h0, theta[0]) + theta[1]
    if batch_norm:
        h1 = tf.contrib.layers.batch_norm(h1, scope='h1', **batch_norm_params)
    h1 = tf.nn.relu(h1)
    h2 = tf.matmul(h1, theta[2]) + theta[3]
    if batch_norm:
        h2 = tf.contrib.layers.batch_norm(h2, scope='h2', **batch_norm_params)
    h2 = tf.nn.relu(h2)
    h3 = tf.matmul(h2, theta[4]) + theta[5]
    return h3


def lfunction(obs, theta, reuse, is_training, scope="lfunction"):
    with tf.variable_scope(scope):
        l = build_NN_two_hidden_layers(obs, theta, reuse, is_training)
        return l


def vec2trimat(vec, dim):
    L = tf.reshape(vec, [-1, dim, dim])
    L = tf.matrix_band_part(L, -1, 0) - tf.matrix_diag(tf.matrix_diag_part(L)) + \
        tf.matrix_diag(tf.exp(tf.matrix_diag_part(L)))
    return L


def ufunction(obs, theta, reuse, is_training, scope="ufunction"):
    with tf.variable_scope(scope):
        act = build_NN_two_hidden_layers(obs, theta, reuse, is_training)
        act = tf.tanh(act)
        return act


def afunction(action, lvalue, uvalue, dimA, scope="afunction"):
    with tf.variable_scope(scope):
        delta = action - uvalue
        L = vec2trimat(lvalue, dimA)

        h1 = tf.reshape(delta, [-1, 1, dimA])
        h1 = tf.batch_matmul(h1, L)  # batch:1:dimA
        h1 = tf.squeeze(h1, [1])  # batch:dimA
        h2 = -tf.constant(0.5) * tf.reduce_sum(h1 * h1, 1)  # batch

        return h2


def qfunction(obs, avalue, theta, reuse, is_training, scope="qfunction"):
    with tf.variable_scope(scope):
        q = build_NN_two_hidden_layers(obs, theta, reuse, is_training)
        q = tf.squeeze(q, [1]) + avalue
        return q

