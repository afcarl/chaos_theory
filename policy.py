import numpy as np
import tensorflow as tf
import tempfile
from gym.spaces import Discrete

from utils import *

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

class DiscretePolicy(Policy):
    """
    A discrete, stochastic policy
    """
    def __init__(self, env):
        super(DiscretePolicy, self).__init__()
        action_space = env.action_space
        self.dU = env.action_space.n
        self.dO = env.reset().shape[0]
        if isinstance(action_space, Discrete):
            self.tf_model = TFDiscrete(self.dU, self.dO)
        else:
            raise NotImplementedError()

    def act(self, obs):
        probs = self.tf_model.probs(obs)
        a = np.random.choice(np.arange(self.dU), p=probs)
        return a

    def grad_act(self, a, obs):
        grad = self.tf_model.grad_probs(a, obs)
        return grad

    def prob_act(self, a, obs):
        probs = self.tf_model.probs(obs)
        return probs[a]

    def set_params(self, paramvec):
        self.tf_model.set_params(paramvec)

    @property
    def params(self):
        return self.tf_model.get_params()

    def copy(self, env):
        pol = DiscretePolicy(env)
        pol.tf_model = self.tf_model.copy()
        return pol


class TFDiscrete(object):
    """
    Tensorflow model for stochastic discrete policies
    """
    def __init__(self, dU, dO):
        self.dU = dU
        self.dO = dO
        self.graph = tf.Graph()
        with self.graph.as_default():
            self._build()
            self._init()

    def _build(self):
        dim_hidden = 10
        self.obs_test = tf.placeholder(tf.float32, (1, self.dO), name='obs_test')

        out = self.obs_test
        with tf.variable_scope('policy') as vs:
            W = tf.get_variable('W', [self.dO, dim_hidden])
            b = tf.get_variable('b', [dim_hidden])
            out = tf.nn.relu(tf.matmul(out, W)+b)

            W = tf.get_variable('Wfin', [dim_hidden, self.dU])
            b = tf.get_variable('bfin', [self.dU])
            out = tf.matmul(out, W)+b
            out = tf.nn.softmax(out)

            params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        self.actions = out

        self.params = params
        self.param_assign_placeholders = [tf.placeholder(tf.float32, param.get_shape()) for param in self.params]
        self.param_assign_ops = [tf.assign(self.params[i], self.param_assign_placeholders[i]) for i in range(len(self.params))]
        self.flattener = ParamFlatten(params)

        act_test = self.actions[0]
        self.grad_prob = [grad_params(act_test[i], self.params) for i in range(self.dU)]

        self.saver = tf.train.Saver()

    def _init(self):
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

    def _run(self, fetches, feeds={}):
        with self.graph.as_default():
            res = self.sess.run(fetches, feed_dict=feeds)
        return res

    def probs(self, obs):
        obs = np.expand_dims(obs, axis=0)
        probs = self._run(self.actions, {self.obs_test: obs})[0]
        return probs

    def get_params(self):
        res = self._run(self.params)
        return self.flattener.pack(res)

    def grad_probs(self, a, obs):
        obs = np.expand_dims(obs, axis=0)
        g = self._run(self.grad_prob[a], {self.obs_test:obs})
        return self.flattener.pack(g)

    def set_params(self, param_vec):
        param_list = self.flattener.unpack(param_vec)
        feed = dict(zip(self.param_assign_placeholders, param_list))
        self._run(self.param_assign_ops, feeds=feed)

    def save(self, fname):
        self.saver.save(self.sess, fname)

    def restore(self, fname):
        self.saver.restore(self.sess, fname)

    def copy(self):
        state = self.__getstate__()
        wts, dU, dO = [state[k] for k in ['wts', 'dU', 'dO']]
        model = TFDiscrete(dU, dO)
        with tempfile.NamedTemporaryFile('w+b', delete=True) as f:
            f.write(wts)
            f.seek(0)
            model.restore(f.name)
        return model
    
    def __getstate__(self):
        with tempfile.NamedTemporaryFile('w+b', delete=True) as f:
            self.save(f.name)
            f.seek(0)
            with open(f.name, 'r') as f2:
                wts = f2.read()
        return {'wts': wts,
                'dU': self.dU,
                'dO': self.dO
        }

    def __setstate__(self, state):
        wts, dU, dO = [state[k] for k in ['wts', 'dU', 'dO']]
        self.__init__(dU, dO)
        with tempfile.NamedTemporaryFile('w+b', delete=True) as f:
            f.write(wts)
            f.seek(0)
            self.restore(f.name)

