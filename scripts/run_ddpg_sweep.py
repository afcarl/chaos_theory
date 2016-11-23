import gym
import tensorflow as tf

from chaos_theory.algorithm import DDPG, two_layer_policy, two_layer_q
from chaos_theory.run.run_algorithm import run_online_algorithm
from chaos_theory.utils.hyper_sweep import run_sweep


def run(env='HalfCheetah-v1', verbose_trial=False, max_iter=100,
        track_tau=0.001,
        discount=0.9, noise_sigma=0.2, actor_lr=1e-4, q_lr=1e-3, hyperparam_string=''):
    tf.reset_default_graph()
    env = gym.make(env)

    policy_arch = two_layer_policy()
    q_network = two_layer_q()

    algorithm = DDPG(env, q_network, policy_arch,
                     discount=discount, noise_sigma=noise_sigma,
                     actor_lr=actor_lr, q_lr=q_lr, track_tau=track_tau)

    run_online_algorithm(env, algorithm, alg_itrs=max_iter, samples_per_update=1,
                         max_length=1000, log_name='ddpg_'+hyperparam_string)


HYPERPARAMS = {
    'discount': [0.9,0.95],
    'noise_sigma': [0.1,0.2,0.3],
    'actor_lr': [1e-3,1e-4],
    'q_lr': [1e-3,1e-4],
    'track_tau': [0.01,0.001]
}

if __name__ == "__main__":
    run_sweep(run, HYPERPARAMS)
