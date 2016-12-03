import logging

import gym
import numpy as np

from chaos_theory.algorithm import DDPG
from chaos_theory.run.run_algorithm import run_online_algorithm

logging.basicConfig(level=logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)
np.random.seed(1)

if __name__ == "__main__":
    #env_name = 'HalfCheetah-v1'
    env_name = 'InvertedPendulum-v1'
    env = gym.make(env_name)
    algorithm = DDPG(env, track_tau=0.001, discount=0.95, q_learning_day=10.)


    log_name = 'ddpg_'+env_name
    log_name = None
    run_online_algorithm(env, algorithm, max_length=1000, samples_per_update=5, verbose_trial=5, log_name=log_name)
