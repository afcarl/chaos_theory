import numpy as np
from utils import discount_rew

def reinforce_grad(pol, trajlist, disc=0.9):
    grad = np.zeros_like(pol.params)
    for traj in trajlist:
        T = traj.T
        for t in range(T):
            r_t = np.sum(discount_rew(traj.rew[t:], gamma=disc))

            grad_logprob = pol.grad_act(traj.act[t], traj.obs[t])
            act_prob = pol.prob_act(traj.act[t], traj.obs[t])
            grad += r_t * (1./act_prob)*grad_logprob
    grad = grad/len(trajlist)
    return grad
