import numpy as np
import random

from chaos_theory.data import ListDataset


class BatchSampler(object):
    def __init__(self, data):
        self.data = data

    def with_replacement(self, batch_size=5, max_itr=float('inf')):
        num_data = len(self.data)
        itr = 0
        while True:
            batch_idx = np.random.randint(0, num_data, size=batch_size)
            yield ListDataset([self.data[idx] for idx in batch_idx])
            if itr < max_itr:
                break


def split_train_test(dataset, train_perc=0.8, shuffle=True):
    random.shuffle(dataset)
    N = len(dataset)
    Ntrain = int(round(train_perc*N))
    train = dataset[:Ntrain]
    test = dataset[Ntrain:]
    return train, test
