import tensorflow as tf

from chaos_theory.algorithm.algorithm import BatchAlgorithm
from chaos_theory.data import ListDataset
from chaos_theory.models.advantage import Advantage
from chaos_theory.models.policy import StochasticPolicyNetwork, NNPolicy
from chaos_theory.utils import assert_shape, make_track_op
from chaos_theory.utils.gym_utils import action_space_dim


class ModifiedTRPO(BatchAlgorithm):
    def __init__(self, env, discount=0.95,
                 normalize_rewards=True, advantage=Advantage(), pol_network=None,
                 lr=5-3, kl_penalty=1e5, track_rate=0.5):
        super(ModifiedTRPO, self).__init__()
        self.advantage = advantage
        self.lr_val = lr
        self.dU = action_space_dim(env.action_space)
        self.normalize_rewards = normalize_rewards
        policy = StochasticPolicyNetwork(env.action_space, env.observation_space,
                                            policy_network=pol_network)
        self.policy_net = policy
        self.policy_net_func = pol_network
        self.dO = policy.dO
        self.dU = policy.dU
        self.discount = discount
        self.kl_penalty=kl_penalty
        self.track_rate=track_rate
        self.__compute_surr()

    def __surr_loss(self, obs_tensor, act_tensor, returns_tensor, batch_size, pol_network):
        # Compute advantages
        advantage = self.advantage.apply(returns_tensor, act_tensor, obs_tensor)
        if self.normalize_rewards:
            avg_advantage = tf.reduce_mean(advantage)
            advantage = advantage/avg_advantage
        assert_shape(advantage, [None])

        # Compute surr loss
        dU = int(act_tensor.get_shape()[-1])
        self.sur_pol_dist = pol_network(obs_tensor, dU, reuse=True)
        log_prob_act = self.sur_pol_dist.log_prob_tensor(act_tensor)
        assert_shape(log_prob_act, [None])

        weighted_logprob = log_prob_act * advantage
        surr_loss = - tf.reduce_sum(weighted_logprob)/batch_size
        return surr_loss

    def __compute_surr(self):
        self.lr = tf.placeholder(tf.float32)
        self.obs_surr= tf.placeholder(tf.float32, [None, self.dO], name='obs_surr')
        self.act_surr = tf.placeholder(tf.float32, [None, self.dU], name='act_surr')
        self.returns_surr = tf.placeholder(tf.float32, [None], name='ret_surr')
        self.batch_size = tf.placeholder(tf.float32, (), name='batch_size')

        self.surr_loss = self.__surr_loss(self.obs_surr, self.act_surr,
                                   self.returns_surr, self.batch_size,
                                   self.policy_net_func)

        with tf.variable_scope('track_pol'):
            self.track_dist = self.policy_net_func(self.obs_surr, self.dU, reuse=False)

        self.track_tau = tf.placeholder(tf.float32, (), name='track')
        self.track_op = make_track_op('track_pol', 'policy', self.track_tau)
        self.kl_step = tf.placeholder(tf.float32, (), name='kl_step')
        kl_obj = self.sur_pol_dist.kl_tensor(self.track_dist)
        kl_loss = self.kl_penalty*tf.nn.relu(kl_obj-self.kl_step)


        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.surr_loss + kl_loss)

        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

    def update(self, samples, inner_itrs=10, kl_step=0.1):
        [samp.apply_discount(self.discount) for samp in samples]

        # Compute advantage function
        self.advantage.update(samples)
        batch = ListDataset(samples)

        # Compute policy gradient
        loss = 0
        for k in range(inner_itrs):
            loss, __ = self.sess.run([self.surr_loss, self.train_op], {
                                                          self.lr: self.lr_val,
                                                          self.obs_surr: batch.concat.obs,
                                                          self.act_surr: batch.concat.act,
                                                          self.batch_size: len(batch),
                                                          self.kl_step: kl_step,
                                                          self.returns_surr: batch.concat.returns})
        self.track_op(self.track_rate)
        return loss

    def get_policy(self):
        return NNPolicy(self.policy_net)