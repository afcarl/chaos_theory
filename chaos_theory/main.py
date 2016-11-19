from chaos_theory.algorithm.reinforce import ReinforceGrad
from chaos_theory.models.advantage import LinearBaseline
from chaos_theory.models.policy import PolicyNetwork, ContinuousPolicy, relu_policy
from chaos_theory.sample import sample, rollout
from utils.utils import print_stats
import numpy as np
import gym
import logging

logging.basicConfig(level=logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)
np.random.seed(0)

#ENV = 'CartPole-v0'
ENV = 'InvertedPendulum-v1'
#ENV = 'HalfCheetah-v1'
#ENV = 'Hopper-v1'

def main():
    """docstring for main"""
    # init_envs(ENV)
    env = gym.make(ENV)
    policy_arch = relu_policy()
    #baseline = LinearBaseline(env.observation_space)
    network = PolicyNetwork(env.action_space, env.observation_space,
            policy_network=policy_arch)
    algorithm = ReinforceGrad(pol_network=network)
    pol = ContinuousPolicy(network)

    disc = 0.90
    for itr in range(10000):
        print '--' * 10, 'itr:', itr
        samps = sample(env, pol, max_length=500, max_samples=20)
        [samp.apply_discount(disc) for samp in samps]

        algorithm.update(samps, 5e-3)

        print_stats(itr, pol, env, samps)
        if itr % 5 == 0:
            samp = rollout(env, pol, max_length=100)
            # print samp.act
            pass

if __name__ == "__main__":
    main()
    # test()
