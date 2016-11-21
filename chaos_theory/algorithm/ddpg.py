import tensorflow as tf

from chaos_theory.algorithm.algorithm import Algorithm
from chaos_theory.data import FIFOBuffer, BatchSampler
from chaos_theory.models.ddpg_networks import CriticQNetwork, TargetQNetwork, compute_sars
from chaos_theory.models.policy import DeterministicPolicyNetwork
from chaos_theory.models.value import create_value_datapoints


class DDPG(Algorithm):
    def __init__(self, obs_space, action_space, q_network, pol_network):
        super(DDPG, self).__init__()
        self.q_network = q_network
        self.obs_space = obs_space
        self.action_space = action_space

        self.replay_buffer = FIFOBuffer(capacity=200)
        self.batch_sampler = BatchSampler(self.replay_buffer)

        # Actor
        self.actor = DeterministicPolicyNetwork(obs_space, action_space, pol_network)
        sess = self.actor.sess

        # Construct Q-networks
        with tf.variable_scope('critic_net'):
            self.critic_network = CriticQNetwork(sess, obs_space, action_space, q_network, self.actor)
        with tf.variable_scope('target_net'):
            self.target_network = TargetQNetwork(sess, obs_space, action_space, q_network, self.critic_network)


    def update(self, samples, **args):
        # Turn trajectories into (s, a, r) tuples
        for traj in samples:
            self.replay_buffer.append_all(compute_sars(traj))

        # Fit q-function
        for batch in self.batch_sampler.with_replacement(batch_size=10):
            batch = self.target_network.compute_returns(batch)
            self.critic_network.train_step(batch, lr=1e-3)

            # Update actor
            self.critic_network.update_policy(batch, lr=1e-3)

        # Track
        self.target_network.track(0.5)
        return float('NaN')

