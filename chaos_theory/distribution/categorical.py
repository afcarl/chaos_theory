import tensorflow as tf
import numpy as np
from chaos_theory.distribution import Distribution
from chaos_theory.utils import linear, assert_shape


class SoftmaxDistribution(Distribution):
    def __init__(self, dU):
        self.dU = dU

    def compute_params_tensor(self, logit):
        probs = tf.nn.softmax(linear(logit, dout=self.dU))
        self.params = [probs]

    def log_prob_tensor(self, actions):
        probs = self.params[0]
        #probs_selected = tf.gather(probs, actions)
        probs_selected = tf.reduce_sum(probs * actions, reduction_indices=[1])
        assert_shape(probs_selected, [None])
        return tf.log(probs_selected)

    def sample(self, N, params):
        if N>1:
            raise NotImplementedError()
        probs = params[0]
        indices = np.random.choice(np.arange(self.dU), size=(N, 1), p=probs[0])
        # Convert to one-hot
        samples = np.zeros((N, self.dU))
        samples[np.arange(N), indices] = 1
        return samples

    def entropy(self, params):
        probs = params[0]
        return -np.sum(params * np.log(params))
