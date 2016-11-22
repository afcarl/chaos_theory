from chaos_theory.algorithm.ddpg import DDPG
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
np.random.seed(0)

#ENV = 'CartPole-v0'
#ENV = 'Pointmass-v1'
ENV = 'InvertedPendulum-v1'
#ENV = 'Reacher-v1'
#ENV = 'Hopper-v1'

MAX_LENGTH = 100

def main():
    """docstring for main"""
    env = gym.make(ENV)
    policy_arch = tanh_deterministic_policy(env.action_space, dim_hidden=4, num_hidden=1)
    q_network = relu_q_fn(num_hidden=1, dim_hidden=10)
    #q_network = pointmass_q_star
    #q_network = linear_q_fn

    algorithm = DDPG(env.observation_space, env.action_space,
                     q_network, policy_arch, discount=0.95, noise_sigma=0.2, track_tau=0.001,
                     actor_lr=1e-4, q_lr=1e-3)
    pol = NNPolicy(algorithm.actor)

    n = 0
    for itr in range(10000):
        print '--' * 10, 'itr:', itr
        #print algorithm.sess.run(algorithm.actor.trainable_vars)
        #print algorithm.sess.run(algorithm.target_network.trainable_vars)

        vars = algorithm.actor.get_vars()
        for var in vars:
            #print var, vars[var]
            pass

        samples = []
        for _ in progress_itr(range(5)):
            pol.reset()
            sample = online_rollout(env, pol, algorithm, max_length=MAX_LENGTH)
            samples.append(sample)
            n += 1
        #samps = sample(env, pol, max_length=MAX_LENGTH, max_samples=10)
        #algorithm.update(samps)
        print_stats(itr, pol, env, samples)
        if itr % 10 == 0 and itr>0:
            rollout(env, pol, max_length=MAX_LENGTH)

if __name__ == "__main__":
    main()
    # test()
