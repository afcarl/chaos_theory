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
                 discount=0.99, noise_sigma=0.1, scale_rew=1.0,
                 actor_lr=1e-4, q_lr=1e-3, track_tau=0.001, weight_decay=1e-2):
        super(DDPG, self).__init__()
        self.q_network = q_network
        self.obs_space = obs_space
        self.action_space = action_space
        self.discount = discount
        self.scale_rew = scale_rew
        self.actor_lr=actor_lr
        self.q_lr=q_lr
        self.track_tau = track_tau
        self.min_batch_size = 1e3

        self.replay_buffer = FIFOBuffer(capacity=5e5)
        self.batch_sampler = BatchSampler(self.replay_buffer)

        # Actor
        self.actor = DeterministicPolicyNetwork(action_space, obs_space, pol_network,
                                                exploration=OUStrategy(action_space, sigma=noise_sigma))
        self.sess = sess = self.actor.sess
        self.target_actor = TargetPolicyNetwork(self.actor, pol_network,
                                                       sess=sess)

        # Construct Q-networks
        self.critic_network = CriticQNetwork(sess, obs_space, action_space, q_network, self.actor, weight_decay=weight_decay)
        self.target_network = TargetQNetwork(sess, obs_space, action_space, q_network, self.target_actor, self.critic_network)

        self.target_network.copy_critic()
        #self.target_actor.copy_actor()
        self.target_actor.track(1.0)


    def update(self, s, a, r, sn, done):
        # Add data to buffer
        self.replay_buffer.append(SARSData(s,a,r*self.scale_rew,sn, 0 if done else 1))
        if len(self.replay_buffer) < self.min_batch_size:
            return

        # Fit q-function and policy
        for batch in self.batch_sampler.with_replacement(batch_size=64, max_itr=1):
            batch = self.target_network.compute_returns(batch, discount=self.discount)
            #print 'RETURNS:', batch.stack.returns[:5]

            self.critic_network.train_step(batch, lr=self.q_lr)
            self.critic_network.update_policy(batch, lr=self.actor_lr)

        # Track critic & policy
        self.target_network.track(self.track_tau)
        self.target_actor.track(self.track_tau)


