from chaos_theory.algorithm.ddpg import DDPG
from chaos_theory.models.policy import tanh_deterministic_policy, NNPolicy
from chaos_theory.models.value import linear_q_fn, relu_q_fn
from chaos_theory.sample import sample, rollout, online_rollout
from chaos_theory.utils.progressbar import progress_itr
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
#ENV = 'Swimmer-v1'
#ENV = 'HalfCheetah-v1'
#ENV = 'Hopper-v1'

MAX_LENGTH = 500

def main():
    """docstring for main"""
    # init_envs(ENV)
    env = gym.make(ENV)
    policy_arch = tanh_deterministic_policy(env.action_space, dim_hidden=10, num_hidden=0)
    q_network = relu_q_fn(num_hidden=1, dim_hidden=10)

    algorithm = DDPG(env.observation_space, env.action_space,
                     q_network, policy_arch, discount=0.9, noise_sigma=0.1)
    pol = NNPolicy(algorithm.actor)

    for itr in range(10000):
        print '--' * 10, 'itr:', itr
        #print algorithm.critic_network.get_vars()

        samples = []
        for _ in progress_itr(range(5)):
            samples.append(online_rollout(env, pol, algorithm, max_length=MAX_LENGTH))
        #samps = sample(env, pol, max_length=MAX_LENGTH, max_samples=10)
        #algorithm.update(samps)
        print_stats(itr, pol, env, samples)
        #if itr % 5 == 0:
        #    samp = rollout(env, pol, max_length=MAX_LENGTH)
        #    pass

if __name__ == "__main__":
    main()
    # test()
