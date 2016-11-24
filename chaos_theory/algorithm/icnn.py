"""
Input Convex Neural Networks
Taken from https://github.com/locuslab/icnn/
"""
import numpy as np
import tensorflow as tf

from chaos_theory.algorithm import OnlineAlgorithm
from chaos_theory.models.policy import Policy

LRELU = 0.01

class ICNN(OnlineAlgorithm, Policy):
    def __init__(self, env):
        super(ICNN, self).__init__()
        dimO = env.observation_space.shape
        dimA = env.observation_space.shape
        self.agent = Agent(dimO, dimA)
        self.init = False

    def get_policy(self):
        return self

    def act(self, obs):
        if not self.init:
            self.agent.reset(obs)
            self.init = True
        return self.agent.act()

    def update(self, s, a, r, sn, done):
        self.agent.observe(r, done, sn)


class Agent:

    def __init__(self, dimO, dimA):
        dimA = list(dimA)
        dimO = list(dimO)
        self.dimA = dimA[0]
        self.dimO = dimO[0]

        tau = 0.001
        discount = 0.95
        l2norm = 0
        learning_rate = 1e-3
        outheta = 0.15
        ousigma = 0.2
        icnn_opt = 'adam'
        rmsize = 1e6
        l1size = 200
        l2size = 200
        self.batch_size = 32
        self.warmup = 1e3

        if icnn_opt == 'adam':
            self.opt = self.adam
        elif icnn_opt == 'bundle_entropy':
            self.opt = self.bundle_entropy
        else:
            raise RuntimeError("Unrecognized ICNN optimizer: "+icnn_opt)

        def entropy(x): #the real concave entropy function
            x_move_reg = tf.clip_by_value((x + 1) / 2, 0.0001, 0.9999)
            pen = x_move_reg * tf.log(x_move_reg) + (1 - x_move_reg) * tf.log(1 - x_move_reg)
            return -tf.reduce_sum(pen, 1)

        # init replay memory
        self.rm = ReplayMemory(rmsize, dimO, dimA)
        # start tf session
        self.sess = tf.Session()

        # create tf computational graph
        self.theta = make_params(dimO[0], dimA[0], l1size, l2size, 'theta')
        self.theta_t, update_t = exponential_moving_averages(self.theta, tau)

        obs = tf.placeholder(tf.float32, [1] + dimO, "obs")
        act_test = tf.placeholder(tf.float32, [1] + dimA, "act")

        # explore
        noise_init = tf.zeros([1] + dimA)
        noise_var = tf.Variable(noise_init, name="noise", trainable=False)
        self.ou_reset = noise_var.assign(noise_init)
        noise = noise_var.assign_sub((outheta) * noise_var - \
                                     tf.random_normal(dimA, stddev=ousigma))
        act_expl = act_test + noise

        # test, single sample q function & gradient for bundle method
        q_test_opt, _, _, _, _ = qfunction(obs, act_test, self.theta, False, False)
        loss_test = -q_test_opt
        act_test_grad = tf.gradients(loss_test, act_test)[0]

        loss_test_entr = -q_test_opt - entropy(act_test)
        act_test_grad_entr = tf.gradients(loss_test_entr, act_test)[0]

        # batched q function & gradient for bundle method
        obs_train2_opt = tf.placeholder(tf.float32, [None] + dimO, "obs_train2_opt")
        act_train2_opt = tf.placeholder(tf.float32, [None] + dimA, "act_train2_opt")

        q_train2_opt, _, _, _, _ = qfunction(obs_train2_opt, act_train2_opt,
                                                  self.theta_t, True, False)
        loss_train2 = -q_train2_opt
        act_train2_grad = tf.gradients(loss_train2, act_train2_opt)[0]

        loss_train2_entr = -q_train2_opt - entropy(act_train2_opt)
        act_train2_grad_entr = tf.gradients(loss_train2_entr, act_train2_opt)[0]

        # training
        obs_train = tf.placeholder(tf.float32, [None] + dimO, "obs_train")
        act_train = tf.placeholder(tf.float32, [None] + dimA, "act_train")
        rew = tf.placeholder(tf.float32, [None], "rew")
        obs_train2 = tf.placeholder(tf.float32, [None] + dimO, "obs_train2")
        act_train2 = tf.placeholder(tf.float32, [None] + dimA, "act_train2")
        term2 = tf.placeholder(tf.bool, [None], "term2")

        q_train, q_train_z1, q_train_z2, q_train_u1, q_train_u2 = qfunction(
            obs_train, act_train, self.theta, True, True)
        q_train_entropy = q_train + entropy(act_train)

        q_train2, _, _, _, _ = qfunction(
            obs_train2, act_train2, self.theta_t, True, True)
        q_train2_entropy = q_train2 + entropy(act_train2)

        # q loss
        if icnn_opt == 'adam':
            q_target = tf.select(term2, rew, rew + discount * q_train2)
            q_target = tf.maximum(q_train - 1., q_target)
            q_target = tf.minimum(q_train + 1., q_target)
            q_target = tf.stop_gradient(q_target)
            td_error = q_train - q_target
        elif icnn_opt == 'bundle_entropy':
            q_target = tf.select(term2, rew, rew + discount * q_train2_entropy)
            q_target = tf.maximum(q_train_entropy - 1., q_target)
            q_target = tf.minimum(q_train_entropy + 1., q_target)
            q_target = tf.stop_gradient(q_target)
            td_error = q_train_entropy - q_target
        ms_td_error = tf.reduce_mean(tf.square(td_error), 0)
        theta = self.theta
        # TODO: Replace with something cleaner, this could easily stop working
        # if the variable names change.
        wd_q = tf.add_n([l2norm * tf.nn.l2_loss(var)
                         if var.name[6] == 'W' else 0. for var in theta])  # weight decay
        loss_q = ms_td_error + wd_q
        # q optimization
        optim_q = tf.train.AdamOptimizer(learning_rate=learning_rate)
        grads_and_vars_q = optim_q.compute_gradients(loss_q)
        optimize_q = optim_q.apply_gradients(grads_and_vars_q)
        with tf.control_dependencies([optimize_q]):
            train_q = tf.group(update_t)


        """
        summary_writer = tf.train.SummaryWriter(os.path.join(FLAGS.outdir, 'board'), self.sess.graph)
        summary_list = []
        if FLAGS.icnn_opt == 'adam':
            summary_list.append(tf.scalar_summary('Qvalue', tf.reduce_mean(q_train)))
        elif FLAGS.icnn_opt == 'bundle_entropy':
            summary_list.append(tf.scalar_summary('Qvalue', tf.reduce_mean(q_train_entropy)))
        summary_list.append(tf.scalar_summary('loss', ms_td_error))
        summary_list.append(tf.scalar_summary('reward', tf.reduce_mean(rew)))
        summary_list.append(tf.scalar_summary('cvx_z1', tf.reduce_mean(q_train_z1)))
        summary_list.append(tf.scalar_summary('cvx_z2', tf.reduce_mean(q_train_z2)))
        summary_list.append(tf.scalar_summary('cvx_z1_pos', tf.reduce_mean(tf.to_float(q_train_z1 > 1e-15))))
        summary_list.append(tf.scalar_summary('cvx_z2_pos', tf.reduce_mean(tf.to_float(q_train_z2 > 1e-15))))
        summary_list.append(tf.scalar_summary('noncvx_u1', tf.reduce_mean(q_train_u1)))
        summary_list.append(tf.scalar_summary('noncvx_u2', tf.reduce_mean(q_train_u2)))
        summary_list.append(tf.scalar_summary('noncvx_u1_pos', tf.reduce_mean(tf.to_float(q_train_u1 > 1e-15))))
        summary_list.append(tf.scalar_summary('noncvx_u2_pos', tf.reduce_mean(tf.to_float(q_train_u2 > 1e-15))))
        """

        # tf functions
        with self.sess.as_default():
            self._reset = Fun([], self.ou_reset)
            self._act_expl = Fun(act_test, act_expl)
            self._train = Fun([obs_train, act_train, rew, obs_train2, act_train2, term2],
                              [train_q, loss_q]) # summary_list, summary_writer

            self._opt_test = Fun([obs, act_test], [loss_test, act_test_grad])
            self._opt_train = Fun([obs_train2_opt, act_train2_opt],
                                  [loss_train2, act_train2_grad])
            self._opt_test_entr = Fun([obs, act_test], [loss_test_entr, act_test_grad_entr])
            self._opt_train_entr = Fun([obs_train2_opt, act_train2_opt],
                                       [loss_train2_entr, act_train2_grad_entr])

        # initialize tf variables
        #self.saver = tf.train.Saver(max_to_keep=1)
        #ckpt = tf.train.latest_checkpoint(FLAGS.outdir + "/tf")
        #if ckpt:
        #    self.saver.restore(self.sess, ckpt)
        #else:
        self.sess.run(tf.initialize_all_variables())

        self.sess.graph.finalize()

        self.t = 0  # global training time (number of observations)

    def bundle_entropy(self, func, obs):
        act = np.ones((obs.shape[0], self.dimA)) * 0.5
        def fg(x):
            value, grad = func(obs, 2 * x - 1)
            grad *= 2
            return value, grad

        act = solveBatch(fg, act)[0]
        act = 2 * act - 1

        return act

    def adam(self, func, obs):
        b1 = 0.9
        b2 = 0.999
        lam = 0.5
        eps = 1e-8
        alpha = 0.01
        nBatch = obs.shape[0]
        act = np.zeros((nBatch, self.dimA))
        m = np.zeros_like(act)
        v = np.zeros_like(act)

        b1t, b2t = 1., 1.
        act_best, a_diff, f_best = [None]*3
        for i in range(10000):
            f, g = func(obs, act)

            if i == 0:
                act_best = act.copy()
                f_best = f.copy()
            else:
                I = (f < f_best)
                act_best[I] = act[I]
                f_best[I] = f[I]

            m = b1 * m + (1. - b1) * g
            v = b2 * v + (1. - b2) * (g * g)
            b1t *= b1
            b2t *= b2
            mhat = m/(1.-b1t)
            vhat = v/(1.-b2t)

            prev_act = act.copy()
            act -= alpha * mhat / (np.sqrt(v) + eps)
            act = np.clip(act, -1, 1)

            a_diff_i = np.mean(np.linalg.norm(act - prev_act, axis=1))
            a_diff = a_diff_i if a_diff is None else lam*a_diff + (1.-lam)*a_diff_i
            # print(a_diff_i, a_diff, np.sum(f))
            if a_diff_i == 0 or a_diff < 1e-2:
                print('  + ADAM took {} iterations'.format(i))
                return act_best

        print('  + Warning: ADAM did not converge.')
        return act_best

    def reset(self, obs):
        self._reset()
        self.observation = obs  # initial observation

    def act(self, test=False):
        print('--- Selecting action, test={}'.format(test))
        obs = np.expand_dims(self.observation, axis=0)
        f = self._opt_test
        act = self.opt(f, obs)

        action = act if test else self._act_expl(act)
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

            if self.t > self.warmup:
                #for i in xrange(FLAGS.iter):
                loss = self.train()

    def train(self):
        obs, act, rew, ob2, term2, info = self.rm.minibatch(size=self.batch_size)
        f = self._opt_train

        print('--- Optimizing for training')
        act2 = self.opt(f, ob2)

        _, loss = self._train(obs, act, rew, ob2, act2, term2,  global_step=self.t)
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
            res = res[: -1]

        return res


def exponential_moving_averages(theta, tau=0.001):
    ema = tf.train.ExponentialMovingAverage(decay=1 - tau)
    update = ema.apply(theta)  # also creates shadow vars
    averages = [ema.average(x) for x in theta]
    return averages, update


def lrelu(x, p=0.01):
    return tf.maximum(x, p * x)

def make_params(dimO, dimA, l1, l2, scope):
    with tf.variable_scope(scope):
        normal_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.1)
        return [tf.get_variable(name='Wx0', shape=[dimO, l1], initializer=normal_init),
                tf.get_variable(name='Wx1', shape=[l1, l2], initializer=normal_init),
                tf.get_variable(name='bx0', shape=[l1], initializer=tf.constant_initializer(0.0)),
                tf.get_variable(name='bx1', shape=[l2], initializer=tf.constant_initializer(0.0)),
                # 4
                tf.get_variable(name='Wzu1', shape=[l1, l1], initializer=normal_init),
                tf.get_variable(name='Wzu2', shape=[l2, l2], initializer=normal_init),
                tf.get_variable(name='Wz1', shape=[l1, l2], initializer=normal_init),
                tf.get_variable(name='Wz2', shape=[l2, 1], initializer=normal_init),
                tf.get_variable(name='bz1', shape=[l1], initializer=tf.constant_initializer(1.0)),
                tf.get_variable(name='bz2', shape=[l2], initializer=tf.constant_initializer(1.0)),
                # 10
                tf.get_variable(name='Wyu0', shape=[dimO, dimA], initializer=normal_init),
                tf.get_variable(name='Wyu1', shape=[l1, dimA], initializer=normal_init),
                tf.get_variable(name='Wyu2', shape=[l2, dimA], initializer=normal_init),
                tf.get_variable(name='Wy0', shape=[dimA, l1], initializer=normal_init),
                tf.get_variable(name='Wy1', shape=[dimA, l2], initializer=normal_init),
                tf.get_variable(name='Wy2', shape=[dimA, 1], initializer=normal_init),
                tf.get_variable(name='by0', shape=[dimA], initializer=tf.constant_initializer(1.0)),
                tf.get_variable(name='by1', shape=[dimA], initializer=tf.constant_initializer(1.0)),
                tf.get_variable(name='by2', shape=[dimA], initializer=tf.constant_initializer(1.0)),
                # 19
                tf.get_variable(name='Wu0', shape=[dimO, l1], initializer=normal_init),
                tf.get_variable(name='Wu1', shape=[l1, l2], initializer=normal_init),
                tf.get_variable(name='Wu2', shape=[l2, 1], initializer=normal_init),
                # 22
                tf.get_variable(name='b0', shape=[l1], initializer=tf.constant_initializer(0.0)),
                tf.get_variable(name='b1', shape=[l2], initializer=tf.constant_initializer(0.0)),
                tf.get_variable(name='b2', shape=[1], initializer=tf.constant_initializer(0.0)),
                ]


def qfunction(obs, act, theta, reuse, is_training, name="qfunction", batch_norm=True):
    batch_norm_params = {'is_training': is_training, 'decay': 0.999, 'epsilon': 1e-3,
                         'updates_collections': None, 'reuse': reuse}

    with tf.variable_op_scope([obs, act], name, name):
        u0 = tf.identity(obs)
        y = tf.identity(act)

        u1 = tf.matmul(u0, theta[0]) + theta[2]
        u1 = tf.nn.relu(u1)
        u2 = tf.matmul(u1, theta[1]) + theta[3]
        if batch_norm:
            u2 = tf.nn.relu(tf.contrib.layers.batch_norm(u2, scope='u2', **batch_norm_params))
        else:
            u2 = tf.nn.relu(u2)

        z1 = tf.matmul((tf.matmul(u0, theta[10]) + theta[16]) * y, theta[13])
        z1 = z1 + tf.matmul(u0, theta[19]) + theta[22]
        z1 = lrelu(z1, LRELU)

        z2 = tf.matmul(tf.nn.relu(tf.matmul(u1, theta[4]) + theta[8]) * z1, tf.abs(theta[6]))
        z2 = z2 + tf.matmul((tf.matmul(u1, theta[11]) + theta[17]) * y, theta[14])
        z2 = z2 + tf.matmul(u1, theta[20]) + theta[23]
        if batch_norm:
            z2 = lrelu(tf.contrib.layers.batch_norm(z2, scope='z2', **batch_norm_params), LRELU)
        else:
            z2 = lrelu(z2, LRELU)

        z3 = tf.matmul(tf.nn.relu(tf.matmul(u2, theta[5]) + theta[9]) * z2, tf.abs(theta[7]))
        z3 = z3 + tf.matmul((tf.matmul(u2, theta[12]) + theta[18]) * y, theta[15])
        z3 = z3 + tf.matmul(u2, theta[21]) + theta[24]
        z3 = -tf.squeeze(z3, [1], name='z3')

        return z3, z1, z2, u1, u2


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

def logistic(x):
    return 1. / (1. + np.exp(-x))

def logexp1p(x):
    """ Numerically stable log(1+exp(x))"""
    y = np.zeros_like(x)
    I = x>1
    y[I] = np.log1p(np.exp(-x[I]))+x[I]
    y[~I] = np.log1p(np.exp(x[~I]))
    return y

def proj_newton_logistic(A,b,lam0=None, line_search=True):
    """ minimize_{lam>=0, sum(lam)=1} -(A*1 + b)^T*lam + sum(log(1+exp(A^T*lam)))"""
    n = A.shape[0]
    c = np.sum(A,axis=1) + b
    e = np.ones(n)

    eps = 1e-12
    ALPHA = 1e-5
    BETA = 0.5

    if lam0 is None:
        lam = np.ones(n)/n
    else:
        lam = lam0.copy()

    for i in range(20):
        # compute gradient and Hessian of objective
        ATlam = A.T.dot(lam)
        z = 1/(1+np.exp(-ATlam))
        f = -c.dot(lam) + np.sum(logexp1p(ATlam))
        g = -c + A.dot(z)
        H = (A*(z*(1-z))).dot(A.T)

        # change of variables
        i = np.argmax(lam)
        y = lam.copy()
        y[i] = 1
        e[i] = 0

        g0 = g - e*g[i]
        H0 = H - np.outer(e,H[:,i]) - np.outer(H[:,i],e) + H[i,i]*np.outer(e,e)

        # compute bound set and Hessian of free set
        I = (y <= eps) & (g0 > 0)
        I[i] = True
        if np.linalg.norm(g0[~I]) < 1e-10:
            return lam
        d = np.zeros(n)
        H0_ = H0[~I,:][:,~I]
        try:
            d[~I] = np.linalg.solve(H0_, -g0[~I])
        except:
            # print('\n=== A\n\n', A)
            # print('\n=== H\n\n', H)
            # print('\n=== H0\n\n', H0)
            # print('\n=== H0_\n\n', H0_)
            # print('\n=== z\n\n', z)
            # print('\n=== iter: {}\n\n'.format(i))
            break
        # line search
        t = min(1. / np.max(abs(d)), 1.)
        for _ in range(10):
            y_n = np.maximum(y + t*d,0)
            y_n[i] = 1
            lam_n = y_n.copy()
            lam_n[i] = 1.-e.dot(y_n)
            if lam_n[i] >= 0:
                if line_search:
                    fn = -c.dot(lam_n) + np.sum(logexp1p(A.T.dot(lam_n)))
                    if fn < f + t*ALPHA*d.dot(g0):
                        break
                else:
                    break
            if max(t * abs(d)) < 1e-10:
                return lam_n
            t *= BETA

        e[i] = 1.
        lam = lam_n.copy()
    return lam

def solveBatch(fg, initXs, nIter=5, callback=None):
    bsize = initXs.shape[0]
    A = [[] for i in range(bsize)]
    b = [[] for i in range(bsize)]
    xs = [[] for i in range(bsize)]
    lam = [None]*bsize

    x = initXs

    finished = []
    nIters = [nIter]*bsize

    finished = set()

    for t in range(nIter):
        fi, gi = fg(x)
        Ai = gi
        bi = fi - np.sum(gi * x, axis=1)
        if callback is not None:
            callback(t, fi)

        for u in range(bsize):
            if u in finished:
                continue

            A[u].append(Ai[u])
            b[u].append(bi[u])
            xs[u].append(np.copy(x[u]))

            prev_x = x[u].copy()
            if len(A[u]) > 1:
                lam[u] = proj_newton_logistic(np.array(A[u]), np.array(b[u]), None)
                x[u] = 1/(1+np.exp(np.array(A[u]).T.dot(lam[u])))
                x[u] = np.clip(x[u], 0.03, 0.97)

            else:
                lam[u] = np.array([1])
                x[u] = 1/(1+np.exp(A[u][0]))
                x[u] = np.clip(x[u], 0.03, 0.97)

            if max(abs((prev_x - x[u]))) < 1e-6:
                finished.add(u)

            A[u] = [y for i,y in enumerate(A[u]) if lam[u][i] > 0]
            b[u] = [y for i,y in enumerate(b[u]) if lam[u][i] > 0]
            xs[u] = [y for i,y in enumerate(xs[u]) if lam[u][i] > 0]
            lam[u] = lam[u][lam[u] > 0]

        if len(finished) == bsize:
            return x, A, b, lam, xs, nIters

    return x, A, b, lam, xs, nIters