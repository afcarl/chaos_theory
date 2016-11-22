from chaos_theory.algorithm.reinforce import ReinforceGrad
from chaos_theory.models.advantage import LinearBaseline
from chaos_theory.models.policy import StochasticPolicyNetwork, NNPolicy, relu_gaussian_policy, linear_softmax_policy
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
#ENV = 'InvertedPendulum-v1'
ENV = 'Pointmass-v1'
#ENV = 'HalfCheetah-v1'
#ENV = 'Hopper-v1'

MAX_LENGTH = 100

def main():
    """docstring for main"""
    #logger = TBLogger('reinforce', {'rew', 'len'})
    env = gym.make(ENV)
    if isinstance(env.action_space, gym.spaces.Discrete):
        policy_arch = linear_softmax_policy()
    else:
        policy_arch = relu_gaussian_policy()
    #baseline = LinearBaseline(env.observation_space)
    network = StochasticPolicyNetwork(env.action_space, env.observation_space,
                                      policy_network=policy_arch)
    algorithm = ReinforceGrad(pol_network=network)
    pol = NNPolicy(network)

    disc = 0.90
    n = 0
    for itr in range(10000):
        print '--' * 10, 'itr:', itr
        samps = sample(env, pol, max_length=MAX_LENGTH, max_samples=10)
        [samp.apply_discount(disc) for samp in samps]

        for samp in samps:
            #logger.log(n, rew=samp.tot_rew, len=samp.T)
            n += 1

        algorithm.update(samps, lr=1e-2)

        print_stats(itr, pol, env, samps)
        if itr % 5 == 0:
            samp = rollout(env, pol, max_length=MAX_LENGTH)

if __name__ == "__main__":
    main()
    # test()
