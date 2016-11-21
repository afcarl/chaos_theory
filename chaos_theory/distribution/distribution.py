
class Distribution(object):
    def log_prob_tensor(self, actions):
        raise NotImplementedError()

    def sample(self, N, params):
        raise NotImplementedError()

    def entropy(self, params):
        raise NotImplementedError()
