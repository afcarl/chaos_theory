from abc import ABCMeta, abstractmethod

import numpy as np

from chaos_theory.utils import discount_value
from chaos_theory.utils.config import FLOAT_X


class Trajectory(object):
    """
    Holds one episode's (s, a, r) tuples in sequential order.
    For batch algorithms.
    """
    def __init__(self, obs, act, rew, info, discount=1.0):
        super(Trajectory, self).__init__()
        self.obs = np.array(obs).astype(np.float)
        self.act = np.array(act)
        self.rew = np.array(rew).astype(np.float)
        self.info = info
        self.T = len(self.rew)
        self.apply_discount(discount)

    def __repr__(self):
        return 'Trajectory(len=%d,r=%f)' % (self.T, sum(self.rew))

    def __len__(self):
        return self.T

    def apply_discount(self, discount=1.0):
        self.discount = discount
        self.disc_rew = discount_value(self.rew, self.discount)

    @property
    def returns(self):
        """ Returns discounted returns """
        return self.disc_rew

    @property
    def tot_rew(self):
        return np.sum(self.rew)


def to_dataset(l):
    if isinstance(l, Dataset):
        return l
    return ListDataset(l)


class Dataset(object):
    """
    List-like object for managing lists of data.
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError()

    @abstractmethod
    def as_list(self):
        raise NotImplementedError()

    @property
    def stack(self):
        """
        Returns a function that stacks all elements of a particular attribute.

        Ex.
        dataset.stack.attr1

        Outputs:
        np.array([ data[0].attr1, data[1].attr1, ... ])

        """
        data_list = self.as_list()
        class Dispatch(object):
            def __init__(self):
                pass
            def __getattr__(self, name):
                values = [getattr(data, name) for data in data_list]
                return np.array(values).astype(FLOAT_X)
        return Dispatch()

    @property
    def concat(self):
        """
        Returns a function that all elements of a particular attribute.

        Ex.
        dataset.concat.attr1

        Outputs:
        np.concatenate([ data[0].attr1, data[1].attr1, ... ])

        """
        data_list = self.as_list()

        class Dispatch(object):
            def __init__(self):
                pass

            def __getattr__(self, name):
                values = [getattr(data, name) for data in data_list]
                return np.concatenate(values).astype(FLOAT_X)

        return Dispatch()


class ListDataset(Dataset):
    """
    >>> from collections import namedtuple
    >>> DataPoint = namedtuple('DataPoint', ['x1', 'x2'])
    >>> dataset = ListDataset([DataPoint([1,1], [0,0]), DataPoint([1,2], [0,1])])
    >>> dataset.stack.x1
    array([[ 1.,  1.],
           [ 1.,  2.]])
    >>> dataset[0:1]
    ListDataset[DataPoint(x1=[1, 1], x2=[0, 0])]
    """
    def __init__(self, l):
        super(ListDataset, self).__init__()
        self.data = l

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return ListDataset(self.data[idx])
        return self.data[idx]

    def as_list(self):
        return self.data

    def __repr__(self):
        return 'ListDataset'+repr(self.data)


class FIFOBuffer(Dataset):
    """
    Limited capacity, first-in first-out buffer

    >>> buf = FIFOBuffer(capacity=2)
    >>> buf.append('a').append('b')
    Buffer['a', 'b']
    >>> buf.append('c')
    Buffer['c', 'b']
    >>> buf.append('d').append('e')
    Buffer['e', 'd']
    """
    def __init__(self, capacity=100):
        super(FIFOBuffer, self).__init__()
        self._buffer = [None] * int(capacity)
        self.C = int(capacity)
        self.active_idx = 0
        self.N = 0

    def append(self, datum):
        self._buffer[self.active_idx] = datum
        self.active_idx = (self.active_idx+1) % self.C
        self.N = min(self.C, self.N+1)
        return self

    def append_all(self, collection):
        for data in collection:
            self.append(data)
        return self

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self._buffer[idx]

    def __repr__(self):
        return 'Buffer'+repr(self._buffer[:self.N])

    def __str__(self):
        return 'Buffer'+str(self._buffer[:self.N])

    def as_list(self):
        return self._buffer[:self.N]

