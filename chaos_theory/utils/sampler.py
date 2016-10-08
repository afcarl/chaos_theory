import numpy as np
import random

class BatchSampler(object):
    def __init__(self, data):
        self.data = data
        self.num_data = len(data)

    def with_replacement(self, batch_size=5):
        while True:
            batch_idx = np.random.randint(0, self.num_data, size=batch_size)
            yield [self.data[idx] for idx in batch_idx]


def split_train_test(dataset, train_perc=0.8, shuffle=True):
    random.shuffle(dataset)
    N = len(dataset)
    Ntrain = int(round(train_perc*N))
    train = dataset[:Ntrain]
    test = dataset[Ntrain:]
    return train, test