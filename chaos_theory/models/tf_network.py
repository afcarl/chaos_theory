import tensorflow as tf
from abc import ABCMeta, abstractmethod
import logging

from chaos_theory.data import BatchSampler
from chaos_theory.utils import get_wt_string, restore_wt_string

LOGGER = logging.getLogger(__name__)

class TFContext(object):
    """
    Helper (like TFNet) which can manage multiple networks living in the same graph
    """
    __metaclass__ = ABCMeta
    def __init__(self, graph=None, sess=None):
        self.__graph = graph if graph else tf.get_default_graph()
        self.__sess = sess if sess else tf.Session(graph=self.__graph)

    def init_network(self):
        with self.__graph.as_default():
            self.saver = tf.train.Saver(tf.trainable_variables())
            self.__sess.run(tf.initialize_all_variables())

    def run(self, fetches, feed_dict=None):
        with self.__graph.as_default():
            results = self.__sess.run(fetches, feed_dict=feed_dict)
        return results

    def get_vars(self):
        with self.__graph.as_default():
            tvars = tf.trainable_variables()
        tvals = self.run(tvars)
        return {tvars[i].name: tvals[i] for i in range(len(tvars))}

    @property
    def sess(self):
        return self.__sess

    @property
    def graph(self):
        return self.__graph

    def __getstate__(self):
        return {'__wts': get_wt_string(self.saver, self.sess)}

    def __setstate__(self, state):
        restore_wt_string(self.saver, self.sess, state['__wts'])

    def __del__(self):
        self.__sess.close()


class TFNet(object):
    def __init__(self):
        pass

    def fit(self, dataset, heartbeat=100, max_iter=float('inf'), batch_size=10, lr=1e-3):
        sampler = BatchSampler(dataset)
        avg_loss = 0
        for i, batch in enumerate(sampler.with_replacement(batch_size=batch_size)):
            loss = self.train_step(batch, lr=lr)
            avg_loss += loss
            if i % heartbeat == 0 and i > 0:
                LOGGER.debug('Itr %d: Loss %f', i, avg_loss / heartbeat)
                avg_loss = 0
            if i > max_iter:
                break

    @abstractmethod
    def train_step(self, batch, lr):
        raise NotImplementedError()

    def __getstate__(self):
        #TODO: handle fields that are tensors
        raise NotImplementedError()

    def __setstate__(self, state):
        raise NotImplementedError()
