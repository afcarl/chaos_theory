import tensorflow as tf
from abc import ABCMeta, abstractmethod
import logging

from chaos_theory.data import BatchSampler

LOGGER = logging.getLogger(__name__)

class TFNet(object):
    __metaclass__ = ABCMeta

    def __init__(self, **build_args):
        #self.__graph = tf.Graph()
        self.__graph = tf.get_default_graph()
        with self.__graph.as_default():
            self.build_network(**build_args)
        self.__init_network()

    @abstractmethod
    def build_network(self, **kwargs):
        raise NotImplementedError()

    def __init_network(self):
        self.__sess = tf.Session(graph=self.__graph)
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
