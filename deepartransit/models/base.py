import tensorflow as tf


class BaseModel:
    def __init__(self, config):
        self.config = config
        self.init_global_step()

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

    def build(self):
        raise NotImplementedError

    @staticmethod
    def gaussian_likelihood(sigma):
        def gaussian_loss(y_true, y_pred):
            return tf.reduce_mean(0.5*tf.log(sigma) + 0.5*tf.div(tf.square(y_true - y_pred), sigma)) + 1e-6 + 6
        return gaussian_loss


class BaseTrainer:
    def __init__(self, sess, model, data, config, logger):
        self.sess = sess
        self.model = model
        self.data = data
        self.config = config
        self.logger = logger

    def train(self):
        raise NotImplementedError

    def train_epoch(self):
        raise NotImplementedError



