import numpy as np

def reinforce_grad(pol, trajlist, disc=0.9, baseline=None, batch_norm=False):
    """
    Compute policy gradient using REINFORCE

    :param pol: A policy object
    :param trajlist: A list of Trajectory objects
    :param disc: A discount factor between 0 and 1
    """
    avg_reward = np.mean([traj.tot_rew/len(traj) for traj in trajlist])

    grad = np.zeros_like(pol.params)
    for traj in trajlist:
        T = traj.T
        for t in range(T):
            obs_t = traj.obs[t]
            act_t = traj.act[t]

            r_t = traj.returns[t] #np.sum(discount_rew(traj.rew[t:], gamma=disc))
            advantage_t = r_t  # No baseline
            if baseline is not None:
                advantage_t = r_t - baseline.eval(obs_t)
            if batch_norm:
                advantage_t = advantage_t / (avg_reward)

            #grad_act = pol.grad_act(act_t, obs_t)
            grad_log_prob = pol.grad_log_act(act_t, obs_t)
            #prob_act = pol.prob_act(act_t, obs_t)
            #print 'prob_act:', prob_act

            # grad = d logprob/dtheta  * advantage
            grad += advantage_t * grad_log_prob #(1./prob_act)*grad_act
    grad = grad/len(trajlist)
    return grad
