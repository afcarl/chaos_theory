import joblib
import multiprocessing
import numpy as np


class Trajectory(object):
    """docstring for Trajectory"""
    def __init__(self, obs, act, rew, info):
        super(Trajectory, self).__init__()
        self.obs = np.array(obs).astype(np.float)
        self.act = np.array(act)
        self.rew = np.array(rew).astype(np.float)
        self.info = info
        self.T = len(self.rew)

    def __repr__(self):
        return 'Trajectory(len=%d,r=%f)' % (self.T, sum(self.rew))

    @property
    def tot_rew(self):
        return np.sum(self.rew)


def rollout(env, policy, render=True, max_length=1000):
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
def _pool():
    if GLOBAL_POOL:
        return GLOBAL_POOL
    return joblib.Parallel(n_jobs=multiprocessing.cpu_count())

def _thread_id():
    return multiprocessing.current_process()._identity[0]-1


GLOB_ENV = None
GLOB_POL = None
def par_rollout(max_length=1):
    #print 'Sampling Thread:', _thread_id()
    print '.',
    return rollout(GLOB_ENV, GLOB_POL, render=False, max_length=max_length)

def sample_par(env, policy, n=1, max_length=1000):
    global GLOB_ENV, GLOB_POL
    GLOB_POL = policy
    GLOB_ENV = env
    print 'Sampling:'
    with _pool() as parallel:
        res = parallel(joblib.delayed(par_rollout)(max_length=max_length) for i in range(n))
    print 'done'
    return res

def sample(env, policy, n=1, max_length=1000):
    return [rollout(env, policy, render=False, max_length=max_length) for _ in range(n)]



