import numpy as np


def entropy(p):
    return -np.sum(p*np.log(p+1e-10))

def pol_entropy(pol, samples):
    total_ent = 0
    tot_n = 0

    for traj in samples:
        for t in range(len(traj)):
            tot_n += 1
            total_ent += pol.act_entropy(traj.obs[t])
    return total_ent / tot_n

def print_stats(itr, pol, env, samples):
    print '--'*10, 'itr:', itr
    N = len(samples)
    R = np.array([samp.tot_rew for samp in samples])
    print 'Avg Rew:', np.mean(R), '+/-', np.sqrt(np.var(R))

    T = np.array([samp.T for samp in samples]).astype(np.float)
    avg_len = np.mean(T)
    stdev = np.sqrt(np.var(T))
    print 'Avg Len:', avg_len, '+/-', stdev
    print 'Num sam:', len(samples)
    
    #all_acts = np.concatenate([samp.act for samp in samples])
    ent = pol_entropy(pol, samples)
    print 'Pol Entropy:', ent
    print 'Perplexity:', 2**ent
    

def discount_rew(rew, gamma=0.99):
    """
    >>> r = np.ones(5)
    >>> discount_rew(r, gamma=0.5).tolist()
    [1.0, 0.5, 0.25, 0.125, 0.0625]
    """
    T = rew.shape[0]
    new_rew = np.zeros_like(rew)
    for i in range(T):
        new_rew[i] = rew[i]*gamma**i
    return new_rew


def discount_value(rew, gamma=0.99):
    values = np.zeros_like(rew)
    for t in range(len(rew)):
        values[t] = np.sum(discount_rew(rew[t:], gamma=gamma))
    return values

def gauss_entropy(sigma):
    dsig = len(sigma)
    detsig = np.prod(sigma)
    detsig *= (2*np.pi * np.e)**dsig
    return 0.5 * np.log(detsig)



        
