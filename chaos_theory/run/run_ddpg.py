import gym
import tensorflow as tf

from chaos_theory.algorithm.ddpg import DDPG
from chaos_theory.models.policy import tanh_deterministic_policy, NNPolicy
from chaos_theory.models.value import relu_q_fn
from chaos_theory.sample import online_rollout, rollout
from chaos_theory.utils import print_stats, TBLogger
from chaos_theory.utils.hyper_sweep import run_sweep
from chaos_theory.utils.progressbar import progress_itr

MAX_LENGTH = 200

def run(env='InvertedPendulum-v1', verbose_trial=False, max_iter=100,
        discount=0.9, noise_sigma=0.2, actor_lr=1e-4, q_lr=1e-3, hyperparam_string=''):
    tf.reset_default_graph()
    logger = TBLogger('ddpg_'+hyperparam_string, {'rew', 'len'})
    env = gym.make(env)
    policy_arch = tanh_deterministic_policy(env.action_space, dim_hidden=10, num_hidden=1)
    q_network = relu_q_fn(num_hidden=1, dim_hidden=10)

    algorithm = DDPG(env.observation_space, env.action_space,
                     q_network, policy_arch, discount=discount, noise_sigma=noise_sigma,
                     actor_lr=actor_lr, q_lr=q_lr, track_tau=0.001)
    pol = NNPolicy(algorithm.actor)

    T = 0
    n = 0
    itr = 0
    while True:
        print '--' * 10, 'itr:', itr
        samples = []
        for _ in progress_itr(range(5)):
            sample = online_rollout(env, pol, algorithm, max_length=MAX_LENGTH)
            logger.log(n, rew=sample.tot_rew, len=sample.T)
            T += sample.T
            n += 1
            samples.append(sample)
        print_stats(itr, pol, env, samples)
        if verbose_trial and itr % 5 == 0 and itr > 0:
            rollout(env, pol, max_length=MAX_LENGTH)

        #if T>max_T:
        #    break
        if itr>max_iter:
            break
        itr += 1


HYPERPARAMS = {
    'discount': [0.9,0.95],
    'noise_sigma': [0.1,0.2,0.3],
    'actor_lr': [1e-3,1e-4],
    'q_lr': [1e-3,1e-4]
}

if __name__ == "__main__":
    run_sweep(run, HYPERPARAMS)
