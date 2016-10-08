import numpy as np
from utils import discount_rew

def reinforce_grad(pol, trajlist, disc=0.9):
    """
    Compute policy gradient using REINFORCE

    :param pol: A policy object
    :param trajlist: A list of Trajectory objects
    :param disc: A discount factor between 0 and 1
    """
    grad = np.zeros_like(pol.params)
    for traj in trajlist:
        T = traj.T
        for t in range(T):
            r_t = np.sum(discount_rew(traj.rew[t:], gamma=disc))
            advantage_t = r_t  # No baseline

            grad_prob = pol.grad_act(traj.act[t], traj.obs[t])
            act_prob = pol.prob_act(traj.act[t], traj.obs[t])

            # grad = d logprob/dtheta  * advantage
            grad += advantage_t * (1./act_prob)*grad_prob
            #grad += advantage_t * grad_prob
    grad = grad/len(trajlist)
    return grad
