from chaos_theory.algorithm.ddpg2 import DDPG2
from chaos_theory.models.policy import tanh_deterministic_policy, NNPolicy
from chaos_theory.models.value import linear_q_fn, relu_q_fn, pointmass_q_star
from chaos_theory.sample import sample, rollout, online_rollout
from chaos_theory.utils.progressbar import progress_itr
from chaos_theory.utils import TBLogger
from utils.utils import print_stats
import numpy as np
import gym
import gym.spaces
import logging

logging.basicConfig(level=logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)
np.random.seed(1)

#ENV = 'Pointmass-v1'
ENV = 'InvertedPendulum-v1'
#ENV = 'Reacher-v1'
#ENV = 'HalfCheetah-v1'

MAX_LENGTH = 100

def main():
    """docstring for main"""
    env = gym.make(ENV)
    algorithm = DDPG2(env, track_tau=0.01, discount=0.9)
    #pol = NNPolicy(algorithm.actor)
    pol = algorithm

    n = 0
    for itr in range(10000):
        print '--' * 10, 'itr:', itr
        samples = []
        for _ in progress_itr(range(5)):
            #pol.reset()
            sample = online_rollout(env, pol, algorithm, max_length=MAX_LENGTH)
            samples.append(sample)
            n += 1
        print_stats(itr, pol, env, samples)
        if itr % 10 == 0 and itr>0:
            rollout(env, pol, max_length=MAX_LENGTH)

if __name__ == "__main__":
    main()
    # test()
