import numpy as np


def entropy(p):
    return -np.sum(p*np.log(p+1e-10))


def pol_entropy(n, acts):
    histo = np.bincount(acts, minlength=n)
    n_act = float(len(acts))
    histo = histo/n_act
    assert len(histo) == n
    return entropy(histo)


def print_stats(env, samples):
    print '--'*10
    N = len(samples)
    R = np.array([samp.tot_rew for samp in samples])
    print 'Avg Rew:', np.mean(R), '+/-', np.sqrt(np.var(R))

    T = np.array([samp.T for samp in samples]).astype(np.float)
    avg_len = np.mean(T)
    stdev = np.sqrt(np.var(T))
    print 'Avg Len:', avg_len, '+/-', stdev
    print 'Num sam:', len(samples)
    
    #all_acts = np.concatenate([samp.act for samp in samples])
    #ent = pol_entropy(env.action_space.n, all_acts)
    #print 'Pol Entropy:', ent
    

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





        
