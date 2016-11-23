import logging

import gym
import gym.spaces
import numpy as np

from chaos_theory.algorithm.trpo import TRPO
from chaos_theory.run.run_algorithm import run_batch_algorithm

logging.basicConfig(level=logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)
np.random.seed(0)

if __name__ == "__main__":
    env = gym.make('InvertedPendulum-v1')

    algorithm = TRPO(env)

    run_batch_algorithm(env, algorithm, samples_per_itr=10, alg_itrs=10000, verbose_trial=-1,
                        max_length=100)
