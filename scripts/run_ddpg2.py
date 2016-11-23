import logging

import gym
import numpy as np

from chaos_theory.algorithm.ddpg2 import DDPG2
from chaos_theory.run.run_algorithm import run_online_algorithm

logging.basicConfig(level=logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)
np.random.seed(1)

if __name__ == "__main__":
    env = gym.make('HalfCheetah-v1')
    algorithm = DDPG2(env, track_tau=0.001, discount=0.9)
    run_online_algorithm(env, algorithm, max_length=500, verbose_trial=10)