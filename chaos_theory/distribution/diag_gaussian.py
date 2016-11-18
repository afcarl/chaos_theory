import numpy as np
import tensorflow as tf

from chaos_theory.utils import linear, assert_shape

class Distribution(object):
    def sample(self, N, *dist_params):
        pass

class DiagGauss(object):
    def __init__(self, dU, mean_clamp=None, min_var=0):
        self.min_var = min_var
        self.dU = dU
        self.mean_clamp = mean_clamp

    def compute_params_tensor(self, logit):
        mu = linear(logit, dout=self.dU, name='mu')
        sigma = tf.exp(linear(logit, dout=self.dU, name='logsig'))
        #params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)
        if self.min_var>0:
            sigma = tf.maximum(self.min_var, sigma)

        if self.mean_clamp:
            mu = tf.nn.tanh(mu) * self.mean_clamp

        dist_params = [mu, sigma]
        self.params = dist_params

    def log_prob_tensor(self, actions):
        mu, sigma = self.params
        #import pdb; pdb.set_trace()
        zs = tf.square(actions - mu) / sigma
        log_sigma = tf.log(sigma)
        logdetsig = tf.reduce_sum(log_sigma, reduction_indices=[-1])
        return - 0.5 * logdetsig \
               - 0.5 * tf.reduce_sum(zs, reduction_indices=[-1]) \
               - 0.5 * self.dU * np.log(2 * np.pi)

    def sample(self, N, mu, sigma):
        samps = np.random.randn(N, self.dU)
        samps = samps*np.sqrt(sigma) + mu
        return samps

    def entropy(self, mu, sigma):
        log_sigma = tf.log(sigma)
        return np.sum(np.sqrt(log_sigma) + np.log(np.sqrt(2 * np.pi * np.e)), axis=-1)

