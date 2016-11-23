"""
NAF implementation, based on ICNN code (https://github.com/locuslab/icnn)
"""
import numpy as np
import tensorflow as tf

from chaos_theory.algorithm.algorithm import OnlineAlgorithm
from chaos_theory.models.policy import Policy


class NAF(OnlineAlgorithm, Policy):
    def __init__(self, env):
        super(NAF, self).__init__()
        dimO = env.observation_space.shape
        dimA = env.observation_space.shape
        self.agent = Agent(dimO, dimA)

    def get_policy(self):
        return self

    def act(self, obs):
        return self.agent.act()

    def update(self, s, a, r, sn, done):
        self.agent.observe(r, done, sn)


class Agent(object):
    def __init__(self, dimO, dimA, tau=0.001, discount=0.95,
                 outheta=0.15, ousigma=0.2, lr=1e-3, l2reg=1e-3,
                 min_replay_size=1e3, replay_size=1e5, batch_size=32):
        dimA = list(dimA)
        dimO = list(dimO)

        l1size = 200
        l2size = 200

        self.batch_size = batch_size
        self.min_replay_size = min_replay_size
        # init replay memory
        self.rm = ReplayMemory(replay_size, dimO, dimA)
        # start tf session
        self.sess = tf.Session()

        is_training = tf.placeholder(tf.bool)

        # create tf computational graph
        self.theta_L = make_params(dimO[0], dimA[0] * dimA[0], l1size, l2size, 'theta_L')
        self.theta_U = make_params(dimO[0], dimA[0], l1size, l2size, 'theta_U')
        self.theta_V = make_params(dimO[0], 1, l1size, l2size, 'theta_V')
        self.theta_Vt, update_Vt = exponential_moving_averages(self.theta_V, tau)

        obs_single = tf.placeholder(tf.float32, [1] + dimO, "obs-single")
        act_test = ufunction(obs_single, self.theta_U, False, is_training)

        # explore
        noise_init = tf.zeros([1] + dimA)
        noise_var = tf.Variable(noise_init, name="noise", trainable=False)
        self.ou_reset = noise_var.assign(noise_init)
        noise = noise_var.assign_sub((outheta) * noise_var - tf.random_normal(dimA, stddev=ousigma))
        act_expl = act_test + noise

        # training
        obs_train = tf.placeholder(tf.float32, [None] + dimO, 'obs_train')
        act_train = tf.placeholder(tf.float32, [None] + dimA, "act_train")
        rew = tf.placeholder(tf.float32, [None], "rew")
        obs2 = tf.placeholder(tf.float32, [None] + dimO, "obs2")
        term2 = tf.placeholder(tf.bool, [None], "term2")
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
        optim_q = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-4)
        grads_and_vars_q = optim_q.compute_gradients(loss_q)
        optimize_q = optim_q.apply_gradients(grads_and_vars_q)
        with tf.control_dependencies([optimize_q]):
            train_q = tf.group(update_Vt)

        """
        summary_writer = tf.train.SummaryWriter(os.path.join(FLAGS.outdir, 'board'), self.sess.graph)
        summary_list = []
        summary_list.append(tf.scalar_summary('Qvalue', tf.reduce_mean(q_train)))
        summary_list.append(tf.scalar_summary('loss', ms_td_error))
        summary_list.append(tf.scalar_summary('reward', tf.reduce_mean(rew)))
        """

        # tf functions
        with self.sess.as_default():
            self._act_test = Fun([obs_single, is_training], act_test)
            self._act_expl = Fun([obs_single, is_training], act_expl)
            self._reset = Fun([], self.ou_reset)
            self._train = Fun([obs_train, act_train, rew, obs2, term2, is_training], [train_q, loss_q])
            #, summary_list, summary_writer)

        # initialize tf variables
        """
        self.saver = tf.train.Saver(max_to_keep=1)
        ckpt = tf.train.latest_checkpoint(FLAGS.outdir + "/tf")
        if ckpt:
            self.saver.restore(self.sess, ckpt)
        else:
            self.sess.run(tf.initialize_all_variables())
        """

        self.sess.graph.finalize()

        self.t = 0  # global training time (number of observations)

    def reset(self, obs):
        self._reset()
        self.observation = obs  # initial observation

    def act(self, test=False):
        obs = np.expand_dims(self.observation, axis=0)
        action = self._act_test(obs, False) if test else self._act_expl(obs, False)
        action = np.clip(action, -1, 1)
        self.action = np.atleast_1d(np.squeeze(action, axis=0))  # TODO: remove this hack
        return self.action

    def observe(self, rew, term, obs2, test=False):

        obs1 = self.observation
        self.observation = obs2

        # train
        if not test:
            self.t = self.t + 1
            self.rm.enqueue(obs1, term, self.action, rew)

            if self.t > self.min_replay_size:
                #for i in xrange(self.iter):
                loss = self.train()

    def train(self):
        obs, act, rew, ob2, term2, info = self.rm.minibatch(size=self.batch_size)
        _, loss = self._train(obs, act, rew, ob2, term2, True, global_step=self.t)
        return loss

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
    L = tf.batch_matrix_band_part(L, -1, 0) - tf.batch_matrix_diag(tf.batch_matrix_diag_part(L)) + \
        tf.batch_matrix_diag(tf.exp(tf.batch_matrix_diag_part(L)))
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


class ReplayMemory:
    def __init__(self, size, dimO, dimA, dtype=np.float32):
        self.size = size
        so = np.concatenate(np.atleast_1d(size, dimO), axis=0)
        sa = np.concatenate(np.atleast_1d(size, dimA), axis=0)
        self.observations = np.empty(so, dtype=dtype)
        self.actions = np.empty(sa, dtype=np.float32)
        self.rewards = np.empty(size, dtype=np.float32)
        self.terminals = np.empty(size, dtype=np.bool)
        self.info = np.empty(size, dtype=object)

        self.n = 0
        self.i = 0

    def reset(self):
        self.n = 0
        self.i = 0

    def enqueue(self, observation, terminal, action, reward, info=None):
        self.observations[self.i, ...] = observation
        self.terminals[self.i] = terminal
        self.actions[self.i, ...] = action
        self.rewards[self.i] = reward
        self.info[self.i, ...] = info
        self.i = (self.i + 1) % self.size
        self.n = min(self.size - 1, self.n + 1)

    def minibatch(self, size):
        indices = np.zeros(size,dtype=np.int)
        for k in range(size):
            invalid = True
            while invalid:
                # sample index ignore wrapping over buffer
                i = np.random.randint(0, self.n - 1)
                # if i-th sample is current one or is terminal: get new index
                if i != self.i and not self.terminals[i]:
                    invalid = False
            indices[k] = i

        o = self.observations[indices, ...]
        a = self.actions[indices]
        r = self.rewards[indices]
        o2 = self.observations[indices + 1, ...]
        t2 = self.terminals[indices + 1]
        info = self.info[indices, ...]

        return o, a, r, o2, t2, info