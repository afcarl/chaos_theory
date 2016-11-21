import tensorflow as tf

from chaos_theory.algorithm.algorithm import Algorithm
from chaos_theory.data import FIFOBuffer, BatchSampler
from chaos_theory.models.ddpg_networks import CriticQNetwork, TargetQNetwork, compute_sars
from chaos_theory.models.policy import DeterministicPolicyNetwork
from chaos_theory.models.value import create_value_datapoints
from chaos_theory.utils.colors import ColorLogger

LOGGER = ColorLogger(__name__)


class DDPG(Algorithm):
    def __init__(self, obs_space, action_space, q_network, pol_network):
        super(DDPG, self).__init__()
        self.q_network = q_network
        self.obs_space = obs_space
        self.action_space = action_space

        self.replay_buffer = FIFOBuffer(capacity=1000)
        self.batch_sampler = BatchSampler(self.replay_buffer)

        # Actor
        self.actor = DeterministicPolicyNetwork(action_space, obs_space, pol_network)
        sess = self.actor.sess

        # Construct Q-networks
        self.critic_network = CriticQNetwork(sess, obs_space, action_space, q_network, self.actor)
        self.target_network = TargetQNetwork(sess, obs_space, action_space, q_network, self.critic_network)
        self.target_network.copy_critic()


    def update(self, samples, **args):
        #print self.target_network.get_vars()

        # Turn trajectories into (s, a, r) tuples
        LOGGER.debug('Adding to replay buffer')
        for traj in samples:
            self.replay_buffer.append_all(compute_sars(traj))


        # Fit q-function
        LOGGER.debug('Fitting Q-function and updating policy')
        for batch in self.batch_sampler.with_replacement(batch_size=10, max_itr=1000):
            batch = self.target_network.compute_returns(batch)
            self.critic_network.train_step(batch, lr=1e-2)

            # Update actor
            self.critic_network.update_policy(batch, lr=1e-2)

        # Track
        self.target_network.track(0.0)
        return float('NaN')

