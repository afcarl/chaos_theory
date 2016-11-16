from chaos_theory.models.policy import PolicyNetwork, ContinuousPolicy, relu_policy
from chaos_theory.sample import sample
from utils.utils import print_stats
import numpy as np
import gym
import logging

logging.basicConfig(level=logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)
np.random.seed(0)

# ENV = 'CartPole-v0'
ENV = 'InvertedPendulum-v1'
#ENV = 'Reacher-v1'

def main():
    """docstring for main"""
    # init_envs(ENV)
    env = gym.make(ENV)
    network = PolicyNetwork(env.action_space, env.observation_space)
            #policy_network=relu_policy())
    pol = ContinuousPolicy(network)

    disc = 0.9
    for itr in range(10000):
        print '--' * 10, 'itr:', itr
        print pol.network.get_vars()
        samps = sample(env, pol, max_length=200, max_samples=40)
        [samp.apply_discount(disc) for samp in samps]

        pol.train_step(samps, 1e-2)

        print_stats(itr, pol, env, samps)
        if itr % 2 == 0:
            # samp = rollout(env, pol, max_length=100)
            # print samp.act
            pass

if __name__ == "__main__":
    main()
    # test()
