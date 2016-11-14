import tensorflow as tf
import numpy as np
import logging

from chaos_theory.data import FIFOBuffer
from chaos_theory.models.value import ValueNetwork, linear_value_fn, create_value_datapoints


LOGGER = logging.getLogger(__name__)


class LinearBaseline(ValueNetwork):
    def __init__(self, obs_space):
        super(LinearBaseline, self).__init__(obs_space, linear_value_fn)
        self.buffer_space = 10000
        self.training_data = FIFOBuffer(self.buffer_space)

    def add_to_buffer(self, traj):
        self.training_data.append_all(create_value_datapoints(traj))

    def clear_buffer(self):
        self.training_data = FIFOBuffer(self.buffer_space)

    def train(self, **kwargs):
        return self.fit(self.training_data, **kwargs)
        
    def eval(self, obs):
        obs = np.expand_dims(obs, axis=0)
        return self.run(self.value_pred, {self.obs: obs})[0]
