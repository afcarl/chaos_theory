from policy import *
from sample import *
from policy_grad import *
from utils.utils import print_stats
from models.baseline import LinearBaseline
import numpy as np
import gym
import logging

logging.basicConfig(level=logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)
np.random.seed(0)

#ENV = 'CartPole-v0'
ENV = 'InvertedPendulum-v1'
#ENV = 'Reacher-v1'

def main():
    """docstring for main"""
    #init_envs(ENV)
    env = gym.make(ENV)
    pol = ContinuousPolicy(env)
    baseline = LinearBaseline(env.observation_space.shape[0])
    lr = 5e-2
    momentum = 0.0
    rew_scale = 1.

    lr_sched = {
            #10: 0.1,
            #50: 0.5,
            #100: 0.5,
            #200: 0.5,
            #300: 0.5,
            }

    prev_grad = np.zeros_like(pol.params)
    disc = 0.95

    for itr in range(10000):
        samps = sample(env, pol, max_length=2000, max_samples=20)
        #scale_rew(rew_scale, *samps)

        #baseline.clear_buffer()
        #[baseline.add_to_buffer(samp, discount=disc) for samp in samps]
        #baseline.train(batch_size=20, heartbeat=500, max_iter=1500, lr=5e-2)
        baseline = None
        
        g = reinforce_grad(pol, samps, disc=disc, baseline=baseline, batch_norm=True)
        g = g/np.linalg.norm(g)
        print 'Gradient magnitude:', np.linalg.norm(g)

        prev_grad = g + momentum*prev_grad
        new_params = pol.params + lr*prev_grad
        pol.set_params(new_params)

        print_stats(itr, pol, env, samps)
        if itr%2 == 0:
            samp = rollout(env, pol, max_length=100)
            #print samp.act
            pass

        if itr in lr_sched:
            lr *= lr_sched[itr]
            

def test():
    env = gym.make(ENV)
    pol = DiscretePolicy(env)

    import pickle
    with open('tmp.tmp', 'w') as f:
        pickle.dump(pol.tf_model, f)



    
if __name__ == "__main__":
    main()
    #test()
