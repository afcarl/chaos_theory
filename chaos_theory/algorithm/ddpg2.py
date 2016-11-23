import numpy as np
import tensorflow as tf

from chaos_theory.algorithm.algorithm import OnlineAlgorithm
from chaos_theory.data import FIFOBuffer, BatchSampler
from chaos_theory.models.ddpg_networks import SARSData
from chaos_theory.models.exploration import OUStrategy
from chaos_theory.utils import linear


def two_layer_policy(l1=200, l2=200):
    def policy(obs, dimA, reuse=False):
        with tf.variable_scope('policy', reuse=reuse):
            h1 = tf.nn.relu(linear(obs, dout=l1, name='h1'))
            h2 = tf.nn.relu(linear(h1, dout=l2, name='h2'))
            h3 = tf.identity(linear(h2, dout=dimA), name='h3')
            action = tf.nn.tanh(h3, name='h4-action')
            return action
    return policy


def two_layer_q(l1=200, l2=200):
    def qfunction(obs, act, reuse=False):
        with tf.variable_scope('qfunc', reuse=reuse):
            h1 = tf.nn.relu(linear(obs, dout=l1, name='h1'))
            h1a = tf.concat(1, [h1, act])
            h2 = tf.nn.relu(linear(h1a, dout=l2, name='h2'))
            qs = linear(h2, dout=1)
            q = tf.squeeze(qs, [1], name='h3-q')
            return q
    return qfunction


class DDPG2(OnlineAlgorithm):
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
                 l2_norm=0,
                 ):
        super(DDPG2, self).__init__()
        dimO = env.observation_space.shape[0]
        dimA = env.action_space.shape[0]

        self.min_buffer_size = min_buffer_size
        pl2norm = l2_norm
        l2norm = l2_norm

        # init replay memory
        self.buffer = FIFOBuffer(rbuffer_size)
        # start tf session
        self.sess = tf.Session()
        self.exploration = OUStrategy(env.action_space, theta=noise_theta, sigma=noise_sigma)


        self.obs = tf.placeholder(tf.float32, [None, dimO], "obs")

        # q optimization
        self.act_train = tf.placeholder(tf.float32, [None, dimA], "act_train")
        self.rew = tf.placeholder(tf.float32, [None], "rew")
        self.obs_next = tf.placeholder(tf.float32, [None, dimO], "obs_next")
        self.term2 = tf.placeholder(tf.bool, [None], "term2")

        # policy loss
        with tf.variable_scope('pol') as vs:
            self.act_test = pol_network(self.obs, dimA)
            vs.reuse_variables()
            act_train_policy = pol_network(self.obs, dimA)
        with tf.variable_scope('critic_q'):
            q_train_policy = q_network(self.obs, act_train_policy)

        meanq = tf.reduce_mean(q_train_policy, 0)
        pol_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pol')
        wd_p = tf.add_n([pl2norm * tf.nn.l2_loss(var) for var in pol_vars])  # weight decay
        loss_p = -meanq + wd_p
        self.pol_optimize_op = tf.train.AdamOptimizer(learning_rate=actor_lr, epsilon=1e-4).minimize(loss_p, var_list=pol_vars)

        # q
        with tf.variable_scope('critic_q', reuse=True):
            q_train = q_network(self.obs, self.act_train)
        with tf.variable_scope('target_pol'):
            act2 = pol_network(self.obs_next, dimA)
        with tf.variable_scope('target_q'):
            q2 = q_network(self.obs_next, act2)
        q_target = tf.stop_gradient(tf.select(self.term2, self.rew, self.rew + discount * q2))

        # q loss
        td_error = q_train - q_target
        ms_td_error = tf.reduce_mean(tf.square(td_error), 0)
        q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_q')
        wd_q = tf.add_n([l2norm * tf.nn.l2_loss(var) for var in q_vars])  # weight decay
        loss_q = ms_td_error + wd_q
        self.q_optimize_op = tf.train.AdamOptimizer(learning_rate=q_lr, epsilon=1e-4).minimize(loss_q, var_list=q_vars)

        # Tracking OPs
        self.pol_track_op = make_track_op('target_pol', 'pol', track_tau)
        self.q_track_op= make_track_op('target_q', 'critic_q', track_tau)

        self.sess.run(tf.initialize_all_variables())

    def update(self, s, a, r, sn, done):
        self.buffer.append(SARSData(s,a,r, sn, 0 if done else 1))

        if len(self.buffer) < self.min_buffer_size:
            return

        for batch in BatchSampler(self.buffer).with_replacement(batch_size=32, max_itr=1):
            obs, act, rew, ob2, term2 = (batch.stack.obs, batch.stack.act, batch.stack.rew,
                                         batch.stack.obs_next, batch.stack.term_mask)
            term2 = (1.0-term2).astype(np.bool)

            self.sess.run(self.pol_optimize_op, {self.obs: obs})
            self.sess.run(self.q_optimize_op, {self.obs: obs, self.act_train: act, self.rew: rew,
                                               self.obs_next: ob2, self.term2: term2})

            self.sess.run(self.q_track_op)
            self.sess.run(self.pol_track_op)

    def reset(self):
        self.exploration.reset()

    def act(self, obs):
        obs = np.expand_dims(obs, axis=0)
        act = self.sess.run(self.act_test, feed_dict={self.obs: obs})
        action = self.exploration.add_noise(act)
        self.action = np.atleast_1d(np.squeeze(action, axis=0))  # TODO: remove this hack
        return self.action

    def __del__(self):
        self.sess.close()


def make_track_op(scope1, scope2, track_rate):
    vars1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope1)
    vars2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope2)
    track_ops = [tf.assign(vars1[i], (1-track_rate)*vars1[i] + (track_rate)*vars2[i]) for i in range(len(vars1))]
    return track_ops


