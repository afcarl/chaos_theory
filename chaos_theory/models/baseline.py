import tensorflow as tf

class LinearBaseline(object):
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default_graph():
            self._build_model()

    def add_to_buffer(self, traj):
        pass

    def _build_model(self):

        self.obs = tf.placeholder(tf.float32, [None, self.dim_obs])
        self.value_labels = tf.placeholder(tf.float32, [None, 1])

        with tf.variable_scope('baseline'):
            w = tf.get_variable('wobs', [self.dim_obs, 1])
            b = tf.get_variable('bobs', [1])
            value = tf.matmul(self.obs, w)+b

        loss = tf.reduce_mean(tf.square(self.value_labels-value))
        optimizer = tf.train.GradientDescentOptimizer(0.5)
        self.train_op = optimizer.minimize(loss)

    def _init_model(self):
        self.sess = tf.Session()
        init = tf.initialize_all_variables()
        self.sess.run(init)


    def train(self):
        
