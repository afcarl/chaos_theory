
class BatchAlgorithm(object):
    def __init__(self):
        pass

    def update(self, samples, *args, **kwargs):
        raise NotImplementedError()


class OnlineAlgorithm(object):
    def __init__(self):
        pass

    def update(self, s, a, r, sn, done):
        raise NotImplementedError()
