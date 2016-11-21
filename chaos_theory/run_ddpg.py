from chaos_theory.algorithm.ddpg import DDPG
from chaos_theory.models.policy import linear_deterministic_policy, NNPolicy
from chaos_theory.models.value import linear_q_fn
from chaos_theory.sample import sample, rollout
from utils.utils import print_stats
import numpy as np
import gym
import gym.spaces
import logging

logging.basicConfig(level=logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)
np.random.seed(0)

#ENV = 'CartPole-v0'
ENV = 'InvertedPendulum-v1'
#ENV = 'HalfCheetah-v1'
#ENV = 'Hopper-v1'

MAX_LENGTH = 500

def main():
    """docstring for main"""
    # init_envs(ENV)
    env = gym.make(ENV)
    policy_arch = linear_deterministic_policy()
    q_network = linear_q_fn
    algorithm = DDPG(env.observation_space, env.action_space,
                     q_network, policy_arch, discount=0.95, noise=5e-2)
    pol = NNPolicy(algorithm.actor)

    for itr in range(10000):
        print '--' * 10, 'itr:', itr
        samps = sample(env, pol, max_length=MAX_LENGTH, max_samples=10)
        algorithm.update(samps)
        print_stats(itr, pol, env, samps)
        if itr % 5 == 0:
            samp = rollout(env, pol, max_length=MAX_LENGTH)
            pass

if __name__ == "__main__":
    main()
    # test()
