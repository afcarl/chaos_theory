import logging

import gym
import numpy as np

from chaos_theory.algorithm import NAF
from chaos_theory.run.run_algorithm import run_online_algorithm

logging.basicConfig(level=logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)
np.random.seed(1)

if __name__ == "__main__":
    env = gym.make('InvertedPendulum-v1')
    algorithm = NAF(env)
    run_online_algorithm(env, algorithm, alg_itrs=10000, samples_per_update=5,
                         verbose_trial=-1, max_length=100)
