import logging

import gym
import numpy as np

from chaos_theory.algorithm.ddpg import DDPG
from chaos_theory.algorithm.ddpg2 import two_layer_policy, two_layer_q
from chaos_theory.run.run_algorithm import run_online_algorithm

logging.basicConfig(level=logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)
np.random.seed(1)

if __name__ == "__main__":
    env = gym.make('InvertedPendulum-v1')
    policy_arch = two_layer_policy()
    q_network = two_layer_q()

    algorithm = DDPG(env, q_network, policy_arch, discount=0.9,
                     noise_sigma=0.2, track_tau=0.01,
                     actor_lr=1e-4, q_lr=1e-3, weight_decay=1e-2)

    run_online_algorithm(env, algorithm, alg_itrs=10000, samples_per_update=5,
                         verbose_trial=10, max_length=100)
