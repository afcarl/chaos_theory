import os
import random
import numpy as np
import tensorflow as tf

#from replay_memory import ReplayMemory
from chaos_theory.algorithm.algorithm import OnlineAlgorithm
from chaos_theory.data import FIFOBuffer, BatchSampler

# DDPG Agent
#
from chaos_theory.models.ddpg_networks import SARSData


class DDPG2(OnlineAlgorithm):
    def __init__(self, env,
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
        l1size = 200
        l2size = 200

        # init replay memory
        self.buffer = FIFOBuffer(rbuffer_size)
        # start tf session
        self.sess = tf.Session(config=tf.ConfigProto(
            #inter_op_parallelism_threads=FLAGS.thread,
            log_device_placement=False,
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1)))

        # create tf computational graph
        #
        self.theta_p = theta_p(dimO, dimA, l1size, l2size)
        self.theta_q = theta_q(dimO, dimA, l1size, l2size)
        self.theta_pt, update_pt = exponential_moving_averages(self.theta_p, track_tau)
        self.theta_qt, update_qt = exponential_moving_averages(self.theta_q, track_tau)

        obs = tf.placeholder(tf.float32, [None, dimO], "obs")
        act_test = policy(obs, self.theta_p)

        # explore
        noise_init = tf.zeros([1, dimA])
        noise_var = tf.Variable(noise_init)
        self.ou_reset = noise_var.assign(noise_init)
        noise = noise_var.assign_sub((noise_theta) * noise_var - tf.random_normal([dimA], stddev=noise_sigma))
        act_expl = act_test + noise

        # q optimization
        act_train = tf.placeholder(tf.float32, [None, dimA], "act_train")
        rew = tf.placeholder(tf.float32, [None], "rew")
        obs2 = tf.placeholder(tf.float32, [None, dimO], "obs2")
        term2 = tf.placeholder(tf.bool, [None], "term2")

        # policy loss
        act_train_policy = policy(obs, self.theta_p)
        q_train_policy = qfunction(obs, act_train_policy, self.theta_q)
        meanq = tf.reduce_mean(q_train_policy, 0)
        wd_p = tf.add_n([pl2norm * tf.nn.l2_loss(var) for var in self.theta_p])  # weight decay
        loss_p = -meanq + wd_p
        # policy optimization
        optim_p = tf.train.AdamOptimizer(learning_rate=actor_lr, epsilon=1e-4)
        grads_and_vars_p = optim_p.compute_gradients(loss_p, var_list=self.theta_p)
        optimize_p = optim_p.apply_gradients(grads_and_vars_p)
        with tf.control_dependencies([optimize_p]):
            train_p = tf.group(update_pt)

        # q
        q_train = qfunction(obs, act_train, self.theta_q)
        # q targets
        act2 = policy(obs2, theta=self.theta_pt)
        q2 = qfunction(obs2, act2, theta=self.theta_qt)
        q_target = tf.stop_gradient(tf.select(term2, rew, rew + discount * q2))
        # q_target = tf.stop_gradient(rew + discount * q2)
        # q loss
        td_error = q_train - q_target
        ms_td_error = tf.reduce_mean(tf.square(td_error), 0)
        wd_q = tf.add_n([l2norm * tf.nn.l2_loss(var) for var in self.theta_q])  # weight decay
        loss_q = ms_td_error + wd_q
        # q optimization
        optim_q = tf.train.AdamOptimizer(learning_rate=q_lr, epsilon=1e-4)
        grads_and_vars_q = optim_q.compute_gradients(loss_q, var_list=self.theta_q)
        optimize_q = optim_q.apply_gradients(grads_and_vars_q)
        with tf.control_dependencies([optimize_q]):
            train_q = tf.group(update_qt)

        # tf functions
        with self.sess.as_default():
            self._act_test = Fun(obs, act_test)
            self._act_expl = Fun(obs, act_expl)
            self._reset = Fun([], self.ou_reset)
            self._train = Fun([obs, act_train, rew, obs2, term2], [train_p, train_q, loss_q])#, summary_list, summary_writer)

        self.sess.run(tf.initialize_all_variables())

    def update(self, s, a, r, sn, done):
        self.buffer.append(SARSData(s,a,r, sn, 0 if done else 1))

        if len(self.buffer) < self.min_buffer_size:
            return

        for batch in BatchSampler(self.buffer).with_replacement(batch_size=32, max_itr=1):
            obs, act, rew, ob2, term2 = (batch.stack.obs, batch.stack.act, batch.stack.rew,
                                         batch.stack.obs_next, batch.stack.term_mask)
            term2 = (1-term2).astype(np.bool)
            _, _, loss = self._train(obs, act, rew, ob2, term2)

    def act(self, obs):
        obs = np.expand_dims(obs, axis=0)
        action = self._act_expl(obs)
        action = np.clip(action, -1, 1)
        self.action = np.atleast_1d(np.squeeze(action, axis=0))  # TODO: remove this hack
        return self.action

    def __del__(self):
        self.sess.close()


# Tensorflow utils
#
class Fun:
    """ Creates a python function that maps between inputs and outputs in the computational graph. """

    def __init__(self, inputs, outputs, summary_ops=None, summary_writer=None, session=None):
        self._inputs = inputs if type(inputs) == list else [inputs]
        self._outputs = outputs
        self._summary_op = tf.merge_summary(summary_ops) if type(summary_ops) == list else summary_ops
        self._session = session or tf.get_default_session()
        self._writer = summary_writer

    def __call__(self, *args, **kwargs):
        """
        Arguments:
          **kwargs: input values
          log: if True write summary_ops to summary_writer
          global_step: global_step for summary_writer
        """
        log = kwargs.get('log', False)

        feeds = {}
        for (argpos, arg) in enumerate(args):
            feeds[self._inputs[argpos]] = arg

        out = self._outputs + [self._summary_op] if log else self._outputs
        res = self._session.run(out, feeds)

        if log:
            i = kwargs['global_step']
            self._writer.add_summary(res[-1], global_step=i)
            res = res[:-1]

        return res

def exponential_moving_averages(theta, tau=0.001):
    ema = tf.train.ExponentialMovingAverage(decay=1 - tau)
    update = ema.apply(theta)  # also creates shadow vars
    averages = [ema.average(x) for x in theta]
    return averages, update



def fanin_init(shape, fanin=None):
    fanin = fanin or shape[0]
    v = 1 / np.sqrt(fanin)
    return tf.random_uniform(shape, minval=-v, maxval=v)


def theta_p(dimO, dimA, l1, l2):
    with tf.variable_scope("theta_p"):
        return [tf.Variable(fanin_init([dimO, l1]), name='1w'),
                tf.Variable(fanin_init([l1], dimO), name='1b'),
                tf.Variable(fanin_init([l1, l2]), name='2w'),
                tf.Variable(fanin_init([l2], l1), name='2b'),
                tf.Variable(tf.random_uniform([l2, dimA], -3e-3, 3e-3), name='3w'),
                tf.Variable(tf.random_uniform([dimA], -3e-3, 3e-3), name='3b')]


def policy(obs, theta, name='policy'):
    with tf.variable_op_scope([obs], name, name):
        h0 = tf.identity(obs, name='h0-obs')
        h1 = tf.nn.relu(tf.matmul(h0, theta[0]) + theta[1], name='h1')
        h2 = tf.nn.relu(tf.matmul(h1, theta[2]) + theta[3], name='h2')
        h3 = tf.identity(tf.matmul(h2, theta[4]) + theta[5], name='h3')
        action = tf.nn.tanh(h3, name='h4-action')
        return action


def theta_q(dimO, dimA, l1, l2):
    with tf.variable_scope("theta_q"):
        return [tf.Variable(fanin_init([dimO, l1]), name='1w'),
                tf.Variable(fanin_init([l1], dimO), name='1b'),
                tf.Variable(fanin_init([l1 + dimA, l2]), name='2w'),
                tf.Variable(fanin_init([l2], l1 + dimA), name='2b'),
                tf.Variable(tf.random_uniform([l2, 1], -3e-4, 3e-4), name='3w'),
                tf.Variable(tf.random_uniform([1], -3e-4, 3e-4), name='3b')]


def qfunction(obs, act, theta, name="qfunction"):
    with tf.variable_op_scope([obs, act], name, name):
        h0 = tf.identity(obs, name='h0-obs')
        h0a = tf.identity(act, name='h0-act')
        h1 = tf.nn.relu(tf.matmul(h0, theta[0]) + theta[1], name='h1')
        h1a = tf.concat(1, [h1, act])
        h2 = tf.nn.relu(tf.matmul(h1a, theta[2]) + theta[3], name='h2')
        qs = tf.matmul(h2, theta[4]) + theta[5]
        q = tf.squeeze(qs, [1], name='h3-q')
        return q

