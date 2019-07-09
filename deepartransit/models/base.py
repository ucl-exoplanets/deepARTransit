import os
import shutil
import tensorflow as tf
from timeit import default_timer as timer

class BaseModel:
    def __init__(self, config):
        self.config = config
        self._init_global_step()
        self._init_cur_epoch()
        self._init_best_loss()

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
            print("Model loaded from step {}".format(self.global_step_tensor.eval(sess)))

    def delete_checkpoints(self):
        if os.path.isdir(self.config.checkpoint_dir):
            print("Deleting model checkpoints ...\n".format(self.config.checkpoint_dir))
            shutil.rmtree(self.config.checkpoint_dir)
            shutil.rmtree(self.config.output_dir)
            shutil.rmtree(self.config.summary_dir)
            shutil.rmtree(self.config.plots_dir)
            print('deleted whole checkpoint, output and plots directory')

    def _init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

    # just initialize a tensorflow variable to use it as epoch counter
    def _init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    def _init_best_loss(self):
        with tf.variable_scope('best_loss'):
            self.best_loss_tensor = tf.Variable(1e10, dtype=tf.float32, name='best_loss')

    def build(self):
        raise NotImplementedError

    @staticmethod
    def gaussian_likelihood(sigma):
        def gaussian_loss(y_true, y_pred):
            return tf.reduce_mean(0.5*tf.log(sigma) + 0.5*tf.math.divide(tf.square(y_true - y_pred), sigma)) + 1e-6 + 6
        return gaussian_loss


class BaseTrainer:
    def __init__(self, sess, model, data, config, logger=None):
        self.sess = sess
        self.model = model
        self.data = data
        self.config = config
        self.logger = logger

    def train(self, verbose=True):
        initial_epoch = self.model.global_step_tensor.eval(self.sess)
        if initial_epoch >= self.config.num_epochs:
            print('model already trained for {} epochs (>= {})'.format(initial_epoch, self.config.num_epochs))
            return 0
        t0 = timer()
        t_eval = 0
        for cur_epoch in range(initial_epoch, self.config.num_epochs):
            result = self.train_epoch()
            if (cur_epoch + 1) % int(self.config.freq_eval) == 0:
                t_eval += self.eval_step(verbose)
                if verbose:
                    print('train epoch result:', result)
                if 'adapt_ranges' in self.config and self.config.adapt_ranges and cur_epoch < self.config.num_epochs // 2:
                    self.update_ranges()

            self.sess.run(self.model.increment_cur_epoch_tensor)
        tf = timer()
        print('total training time: {} \nAVG time per epoch: {}'
              .format(tf-t0, (tf-t0)/(self.config.num_epochs-initial_epoch)))
        print('including {}s for {} evaluation steps in total'.format(t_eval,
                                                                      self.config.num_epochs
                                                                      // self.config.freq_eval))

    def train_epoch(self):
        for iteration in range(self.config.num_iter):
            self.train_step()
        cur_it = self.model.global_step_tensor.eval(self.sess)
        self.logger.summarize(cur_it, summaries_dict={})
        # TODO: fill the logger
        self.model.save(self.sess)




