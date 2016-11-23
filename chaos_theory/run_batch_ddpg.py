from chaos_theory.algorithm.ddpg import DDPG, BatchDDPG
from chaos_theory.algorithm.reinforce import ReinforceGrad
from chaos_theory.models.advantage import LinearBaseline
from chaos_theory.models.policy import StochasticPolicyNetwork, NNPolicy, relu_gaussian_policy, linear_softmax_policy, \
    tanh_deterministic_policy
from chaos_theory.models.value import linear_q_fn
from chaos_theory.sample import sample, rollout
from chaos_theory.utils import TBLogger
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
#ENV = 'Pointmass-v1'
#ENV = 'HalfCheetah-v1'
#ENV = 'Hopper-v1'

MAX_LENGTH = 100

def main():
    """docstring for main"""
    #logger = TBLogger('reinforce', {'rew', 'len'})
    env = gym.make(ENV)
    policy_arch = tanh_deterministic_policy(env.action_space, dim_hidden=4, num_hidden=0)
    #q_network = relu_q_fn(num_hidden=1, dim_hidden=10)
    #q_network = pointmass_q_star
    q_network = linear_q_fn

    algorithm = BatchDDPG(env.observation_space, env.action_space,
                     q_network, policy_arch, noise_sigma=0.2,
                     weight_decay=1e-2)
    pol = NNPolicy(algorithm.actor)
    disc=0.95

    for itr in range(10000):
        print '--' * 10, 'itr:', itr
        samps = sample(env, pol, max_length=MAX_LENGTH, max_samples=10)
        [samp.apply_discount(disc) for samp in samps]

        algorithm.update(samps)

        print_stats(itr, pol, env, samps)
        if itr % 5 == 0:
            samp = rollout(env, pol, max_length=MAX_LENGTH)

if __name__ == "__main__":
    main()
    # test()
