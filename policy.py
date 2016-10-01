import numpy as np
import tensorflow as tf
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
    def __init__(self, env):
        super(DiscretePolicy, self).__init__()
        self.action_space = env.action_space
        self.dU = env.action_space.n
        self.dO = env.reset().shape[0]
        if isinstance(env.action_space, Discrete):
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


class TFDiscrete(object):
    def __init__(self, dU, dO):
        self.dU = dU
        self.dO = dO
        self._build()
        self._init()

    def _build(self):
        dim_hidden = 5
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

    def _init(self):
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

    def _run(self, fetches, feeds={}):
        return self.sess.run(fetches, feed_dict=feeds)

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


