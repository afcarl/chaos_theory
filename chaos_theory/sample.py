import joblib
import multiprocessing
import numpy as np
import gym
from chaos_theory.data import Trajectory


def rollout(env, policy, render=True, max_length=float('inf')):
    obs = env.reset()
    done = False
    obs_list = [obs]
    rew_list = []
    act_list = []
    info_list = []
    t = 0
    while not done:
        a = policy.act(obs)
        obs, rew, done, info = env.step(a)
        if render:
            env.render()
        obs_list.append(obs)
        rew_list.append(rew)
        act_list.append(a)
        info_list.append(info)

        t += 1
        if t > max_length:
            break

    traj = Trajectory(obs_list, act_list, rew_list, info_list)
    return traj


GLOBAL_POOL = None
"""
def _pool():
    if GLOBAL_POOL:
        return GLOBAL_POOL
    #return joblib.Parallel(n_jobs=multiprocessing.cpu_count())
    print 'INIT_PAR'*10
    global GLOBAL_POOL
    GLOBAL_POOL = joblib.Parallel(n_jobs=2)
    return GLOBAL_POOL
"""

def _thread_id():
    n = _pool().n_jobs
    thid = multiprocessing.current_process()._identity
    if len(thid) == 0:
        return -999
    else:
        return (thid[0]-1) % n

GLOBAL_ENVS = []
def init_envs(name):
    global GLOBAL_ENVS
    n = _pool().n_jobs
    GLOBAL_ENVS = [gym.make(name) for _ in range(n)] 

def init_pols(p, env):
    # Duplicate the policy
    global GLOBAL_POLS
    n = _pool().n_jobs
    #GLOBAL_POLS = [p.copy(env) for _ in range(n)] 
    from policy import RandomPolicy
    GLOBAL_POLS = [RandomPolicy(env.action_space) for _ in range(n)]


GLOB_POL = None
def par_rollout(max_length=1):
    tid = _thread_id()
    print '.',
    return rollout(GLOBAL_ENVS[tid], GLOBAL_POLS[tid], render=False, max_length=max_length)

def sample_par(env, pol, n=1, max_length=1000):
    init_pols(pol, env)
    print 'Sampling:'
    with _pool() as parallel:
        res = parallel(joblib.delayed(par_rollout)(max_length=max_length) for i in range(n))
    print 'done'
    return res

def sample(env, policy, max_samples=float('inf'), max_length=float('inf')):
    tot_len = 0
    samples = []
    while tot_len < max_length:
        traj = rollout(env, policy, render=False, max_length=max_length)
        tot_len += len(traj)
        samples.append(traj)
        if len(samples) >= max_samples:
            break
    return samples



