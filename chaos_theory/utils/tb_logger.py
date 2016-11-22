import os
import tensorflow as tf

class TBLogger(object):
    def __init__(self, exp_name, keys):
        self.exp_name = exp_name
        self.keys = keys

        self.placeholders = {}
        self.summaries = {}
        for key in keys:
            self.placeholders[key] = tf.placeholder(tf.float32)
            self.summaries[key] = tf.scalar_summary(key, self.placeholders[key])

        self.writer = tf.train.SummaryWriter(os.path.join('logdir', exp_name))
        self.sess = tf.Session()

    def log(self, i, **kwargs):
        feed = {self.placeholders[key]: kwargs[key] for key in kwargs}
        summaries = tf.merge_summary([self.summaries[key] for key in kwargs])
        self.writer.add_summary(self.sess.run(summaries, feed_dict=feed), i)
