from policy import *
from sample import *
from policy_grad import *
from utils.utils import print_stats
from models.baseline import LinearBaseline
import gym
import logging

logging.basicConfig(level=logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)

#ENV = 'CartPole-v0'
ENV = 'CartPole-v0'

def main():
    """docstring for main"""
    init_envs(ENV)
    env = gym.make(ENV)
    pol = DiscretePolicy(env)
    baseline = LinearBaseline(4)
    lr = 1e-2
    momentum = 0.5

    lr_sched = {
            50: 0.5,
            100: 0.5,
            200: 0.5,
            300: 0.5,
            }

    prev_grad = np.zeros_like(pol.params)
    disc = 0.90

    for itr in range(1000):
        samps = sample(env, pol, max_length=200*5)

        baseline.clear_buffer()
        [baseline.add_to_buffer(samp, discount=disc) for samp in samps]
        baseline.train(batch_size=20, heartbeat=500, max_iter=1000)
        
        g = reinforce_grad(pol, samps, disc=disc, baseline=baseline)

        prev_grad = g + momentum*prev_grad
        new_params = pol.params + lr*prev_grad
        pol.set_params(new_params)

        print_stats(env, samps)
        if itr%10 == 0:
            print 'itr:', itr
            #rollout(env, pol, max_length=300)

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
