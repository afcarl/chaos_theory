import tensorflow as tf

from chaos_theory.algorithm.algorithm import OnlineAlgorithm

from chaos_theory.data import FIFOBuffer, BatchSampler
from chaos_theory.models.ddpg_networks import CriticQNetwork, TargetQNetwork, SARSData, TargetPolicyNetwork
from chaos_theory.models.exploration import OUStrategy
from chaos_theory.models.policy import DeterministicPolicyNetwork
from chaos_theory.utils.colors import ColorLogger

LOGGER = ColorLogger(__name__)


class DDPG(OnlineAlgorithm):
    def __init__(self, obs_space, action_space, q_network, pol_network,
                 discount=0.99, noise_sigma=0.1, scale_rew=1.0):
        super(DDPG, self).__init__()
        self.q_network = q_network
        self.obs_space = obs_space
        self.action_space = action_space
        self.discount = discount
        self.scale_rew = scale_rew
        self.min_batch_size = 1e3

        self.replay_buffer = FIFOBuffer(capacity=1e6)
        self.batch_sampler = BatchSampler(self.replay_buffer)

        # Actor
        self.actor = DeterministicPolicyNetwork(action_space, obs_space, pol_network,
                                                exploration=OUStrategy(action_space, sigma=noise_sigma))
        sess = self.actor.sess
        self.target_actor = TargetPolicyNetwork(self.actor, pol_network,
                                                       sess=sess)
        self.target_actor.copy_actor()

        # Construct Q-networks
        self.critic_network = CriticQNetwork(sess, obs_space, action_space, q_network, self.actor, weight_decay=1e-2)
        self.target_network = TargetQNetwork(sess, obs_space, action_space, q_network, self.target_actor, self.critic_network)
        self.target_network.copy_critic()


    def update(self, s, a, r, sn):
        # Add data to buffer
        self.replay_buffer.append(SARSData(s,a,r*self.scale_rew,sn))
        if len(self.replay_buffer) < self.min_batch_size:
            return

        # Fit q-function and policy
        for batch in self.batch_sampler.with_replacement(batch_size=32, max_itr=1):
            batch = self.target_network.compute_returns(batch, discount=self.discount)

            self.critic_network.train_step(batch, lr=1e-3)
            self.critic_network.update_policy(batch, lr=1e-4)

        # Track critic & policy
        self.target_network.track(0.001)
        self.target_actor.track(0.001)
        return float('NaN')
