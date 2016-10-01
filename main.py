from policy import *
from sample import *
from policy_grad import *
from utils import print_stats
import gym

def main():
    """docstring for main"""
    env = gym.make('CartPole-v1')
    #env = gym.make('FrozenLake-v0')
    #pol = RandomPolicy(env.action_space)

    pol = DiscretePolicy(env)
    lr = 1e-2

    lr_sched = {
            100: 0.5,
            200: 0.5,
            300: 0.5,
            400: 0.5,
            }

    for itr in range(500):
        samps = sample(env, pol, n=10, max_length=200)
        g = reinforce_grad(pol, samps, disc=0.9)
        new_params = pol.params + lr*g
        pol.set_params(new_params)

        print_stats(env, samps)
        if itr%10 == 0:
            print 'itr:', itr
            rollout(env, pol, max_length=1000)
            print pol.params

        if itr in lr_sched:
            lr *= lr_sched[itr]
            


    
if __name__ == "__main__":
    main()
