import tensorflow as tf

from chaos_theory.data import ListDataset
from chaos_theory.models.advantage import Advantage
from chaos_theory.utils import assert_shape


class ReinforceGrad():
    def __init__(self, normalize_rewards=True, advantage=Advantage()):
        self.advantage = advantage
        self.normalize_rewards = normalize_rewards

    def set_policy(self, tf_net):
        self.policy_net = tf_net

    def surr_loss(self, obs_tensor, act_tensor, returns_tensor, batch_size, pol_network):
        self.obs_surr = obs_tensor
        self.act_surr = act_tensor
        self.returns_surr = returns_tensor
        self.batch_size = batch_size

        # Compute advantages
        advantage = self.advantage.apply(returns_tensor, act_tensor, obs_tensor)
        if self.normalize_rewards:
            avg_advantage = tf.reduce_mean(advantage)
            advantage = advantage/avg_advantage

        #advantage = tf.Print(advantage, [advantage], summarize=10, message='Advtg')
        assert_shape(advantage, [None])

        # Compute surr loss
        dU = int(act_tensor.get_shape()[-1])
        sur_pol_dist = pol_network(obs_tensor, dU, reuse=True)
        #act_tensor = tf.Print(act_tensor, [act_tensor], summarize=10, message='actions')
        log_prob_act = sur_pol_dist.log_prob_tensor(act_tensor)
        assert_shape(log_prob_act, [None])

        # debugging
        #prob_act = tf.exp(log_prob_act)
        #prob_act = tf.Print(prob_act, [prob_act], summarize=10, message='like')
        #log_prob_act = tf.log(prob_act)
        #log_prob_act = tf.Print(log_prob_act, [log_prob_act], summarize=10, message='Logli')

        weighted_logprob = log_prob_act * advantage
        #weighted_logprob = tf.Print(weighted_logprob, [weighted_logprob], summarize=10, message='WeightLogli')
        self.surr_loss = - tf.reduce_sum(weighted_logprob)/batch_size
        return self.surr_loss

    def update(self, samples, lr):
        self.advantage.update(samples)
        batch = ListDataset(samples)
        return self.policy_net.run([self.surr_loss, self.policy_net.train_op], {self.policy_net.lr: lr,
                                                          self.obs_surr: batch.concat.obs,
                                                          self.act_surr: batch.concat.act,
                                                          self.batch_size: len(batch),
                                                          self.returns_surr: batch.concat.returns})[0]
