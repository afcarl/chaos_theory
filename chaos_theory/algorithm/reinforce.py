import tensorflow as tf

from chaos_theory.algorithm.algorithm import BatchAlgorithm
from chaos_theory.data import ListDataset
from chaos_theory.models.advantage import Advantage
from chaos_theory.models.policy import StochasticPolicyNetwork
from chaos_theory.utils import assert_shape


class ReinforceGrad(BatchAlgorithm):
    def __init__(self, env, discount=0.95,
                 normalize_rewards=True, advantage=Advantage(), pol_network=None,
                 lr=5-3):
        super(ReinforceGrad, self).__init__()
        self.advantage = advantage
        self.normalize_rewards = normalize_rewards
        policy = StochasticPolicyNetwork(env.action_space, env.observation_space,
                                            policy_network=pol_network)
        self.policy_net = policy
        self.dO = policy.dO
        self.dU = policy.dU
        self.discount = discount
        self.__compute_surr()

    def surr_loss(self, obs_tensor, act_tensor, returns_tensor, batch_size, pol_network):
        # Compute advantages
        advantage = self.advantage.apply(returns_tensor, act_tensor, obs_tensor)
        if self.normalize_rewards:
            avg_advantage = tf.reduce_mean(advantage)
            advantage = advantage/avg_advantage
        assert_shape(advantage, [None])

        # Compute surr loss
        dU = int(act_tensor.get_shape()[-1])
        sur_pol_dist = pol_network(obs_tensor, dU, reuse=True)
        log_prob_act = sur_pol_dist.log_prob_tensor(act_tensor)
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

        self.surr_loss = self.surr_loss(self.obs_surr, self.act_surr,
                                   self.returns_surr, self.batch_size,
                                   self.policy_net.policy_network)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.surr_loss)

        self.sess = self.policy_net.sess
        self.sess.run(tf.initialize_all_variables())

    def update(self, samples, **args):
        [samp.apply_discount(self.discount) for samp in samples]

        # Compute advantage function
        self.advantage.update(samples)
        batch = ListDataset(samples)

        # Compute policy gradient
        loss, __ = self.sess.run([self.surr_loss, self.train_op], {
                                                          self.lr: lr,
                                                          self.obs_surr: batch.concat.obs,
                                                          self.act_surr: batch.concat.act,
                                                          self.batch_size: len(batch),
                                                          self.returns_surr: batch.concat.returns})
        return loss

    def get_policy(self):
        return NNPolicy()