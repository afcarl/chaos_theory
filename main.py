from policy import *
from sample import *
from policy_grad import *
from utils import print_stats
import gym

ENV = 'CartPole-v0'

def main():
    """docstring for main"""
    init_envs(ENV)
    env = gym.make(ENV)
    pol = DiscretePolicy(env)
    lr = 1e-3
    momentum = 0.9

    lr_sched = {
            50: 0.5,
            100: 0.5,
            200: 0.5,
            300: 0.5,
            }

    prev_grad = np.zeros_like(pol.params)
    for itr in range(1000):
        samps = sample(env, pol, n=10, max_length=300)
        g = reinforce_grad(pol, samps, disc=0.95)

        prev_grad = g + momentum*prev_grad
        new_params = pol.params + lr*prev_grad
        pol.set_params(new_params)

        print_stats(env, samps)
        if itr%10 == 0:
            print 'itr:', itr
            rollout(env, pol, max_length=300)

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
