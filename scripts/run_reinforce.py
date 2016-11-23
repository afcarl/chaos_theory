import logging

import gym
import gym.spaces
import numpy as np

from chaos_theory.algorithm.reinforce import ReinforceGrad
from chaos_theory.models.network_defs import linear_softmax_policy, relu_gaussian_policy
from chaos_theory.run.run_algorithm import run_batch_algorithm

logging.basicConfig(level=logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)
np.random.seed(0)

if __name__ == "__main__":
    env = gym.make('InvertedPendulum-v1')

    if isinstance(env.action_space, gym.spaces.Discrete):
        policy_arch = linear_softmax_policy()
    else:
        policy_arch = relu_gaussian_policy()

    # baseline = LinearBaseline(env.observation_space)
    algorithm = ReinforceGrad(env, discount=0.9, pol_network=policy_arch, lr=5e-3)

    run_batch_algorithm(env, algorithm, samples_per_itr=10, alg_itrs=10000, verbose_trial=5,
                        max_length=100)
