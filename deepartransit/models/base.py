import tensorflow as tf


class BaseModel:
    def __init__(self, config):
        self.config = config
        self.init_global_step()
        self.init_cur_epoch()

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=5)

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, sess):
        print('Saving model...')
        self.saver.save(sess, self.config.checkpoint_dir, self.global_step_tensor)
        print('model saved')

    def load(self, sess):
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded from {}".format(self.global_step_tensor))

    def init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

    # just initialize a tensorflow variable to use it as epoch counter
    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    def build(self):
        raise NotImplementedError

    @staticmethod
    def gaussian_likelihood(sigma):
        def gaussian_loss(y_true, y_pred):
            return tf.reduce_mean(0.5*tf.log(sigma) + 0.5*tf.div(tf.square(y_true - y_pred), sigma)) + 1e-6 + 6
        return gaussian_loss


class BaseTrainer:
    """
    Basic config:
        - num_iter
        - num_epochs
    """
    def __init__(self, sess, model, data, config, logger=None):
        self.sess = sess
        self.model = model
        self.data = data
        self.config = config
        self.logger = logger

    def train(self, verbose=False):
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs + 1, 1):
            result = self.train_epoch()
            self.sess.run(self.model.increment_cur_epoch_tensor)
            if verbose:
                print('curr epoch :', self.model.cur_epoch_tensor.eval(self.sess))
                print('train result:', result)

    def train_epoch(self):
        for iteration in range(self.config.num_iter):
            self.train_step()
        cur_it = self.model.global_step_tensor.eval(self.sess)
        #self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def train_step(self):
        raise NotImplementedError

