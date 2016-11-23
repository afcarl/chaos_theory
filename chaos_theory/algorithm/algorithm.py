
class BatchAlgorithm(object):
    def __init__(self):
        pass

    def get_policy(self):
        raise NotImplementedError()

    def update(self, samples):
        raise NotImplementedError()


class OnlineAlgorithm(object):
    def __init__(self):
        pass

    def get_policy(self):
        raise NotImplementedError()

    def update(self, s, a, r, sn, done):
        raise NotImplementedError()
