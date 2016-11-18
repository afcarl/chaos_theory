from abc import ABCMeta, abstractmethod

import numpy as np

from chaos_theory.utils import discount_value
from chaos_theory.utils.config import FLOAT_X


class Trajectory(object):
    """docstring for Trajectory"""

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
        return self.disc_rew

    @property
    def tot_rew(self):
        return np.sum(self.rew)


class Dataset(object):
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
        data_list = self.as_list()
        class Dispatch(object):
            def __init__(self):
                pass
            def __getattr__(self, name):
                values = [getattr(data, name) for data in data_list]
                import pdb; pdb.set_trace()
                return np.array(values).astype(FLOAT_X)
        return Dispatch()

    @property
    def concat(self):
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
    Buffer['b', 'c']
    >>> buf._buffer
    ['a', 'b', 'c']
    >>> buf.append('d').append('e')
    Buffer['d', 'e']
    >>> buf._buffer
    ['d', 'e']
    """
    def __init__(self, capacity=100, overflow_factor=2):
        super(FIFOBuffer, self).__init__()
        self._buffer = []
        self.C = capacity
        self.active_idx = 0
        self.O = capacity*overflow_factor

    def append(self, datum):
        self._buffer.append(datum)

        if len(self._buffer) > self.C:
            self.active_idx += 1

        # Reset data list if overflowing
        if len(self._buffer) > self.O:
            self._buffer = self._buffer[-self.C:]
            self.active_idx = 0

        return self

    def append_all(self, collection):
        for data in collection:
            self.append(data)
        return self

    def __len__(self):
        return min(self.C, len(self._buffer))

    def __getitem__(self, idx):
        return self._buffer[self.active_idx+idx]

    def __repr__(self):
        return 'Buffer'+repr(self._buffer[self.active_idx:])

    def __str__(self):
        return 'Buffer'+str(self._buffer[self.active_idx:])

    def as_list(self):
        return self._buffer[self.active_idx:]

