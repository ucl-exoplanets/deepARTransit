import os
import shutil
import numpy as np
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
            shutil.rmtree(self.config.checkpoint_dir, ignore_errors=True)
            shutil.rmtree(self.config.output_dir, ignore_errors=True)
            shutil.rmtree(self.config.summary_dir, ignore_errors=True)
            shutil.rmtree(self.config.plots_dir, ignore_errors=True)
            print('deleted whole checkpoint, output and plots directory')

    def _init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

    def _init_learning_rate(self):
        with tf.variable_scope('learning_rate'):
            if ("starter_learning_rate" in self.config and "end_learning_rate" in self.config
                    and "power" in self.config and "decay_steps" in self.config):
                self.learning_rate_tensor = tf.train.polynomial_decay(self.config.starter_learning_rate, self.global_step_tensor,
                                                               self.config.decay_steps, self.config.end_learning_rate, self.config.power)
            else:
                self.learning_rate_tensor = tf.Variable(self.config.learning_rate, trainable=False, name='learning_rate')


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
    def gaussian_likelihood(sigma, weights=1.0):
        def gaussian_loss(y_true, y_pred):
            return tf.reduce_mean(tf.math.multiply(weights, 0.5*tf.log(sigma) + 0.5*tf.math.divide(tf.square(y_true - y_pred), sigma))) + 1e-6 + 6
            #@return tf.losses.compute_weighted_loss(losses, weights)
        return gaussian_loss


class BaseTrainer:
    def __init__(self, sess, model, data, config, logger=None):
        self.sess = sess
        self.model = model
        self.data = data
        self.config = config
        self.logger = logger
        self.best_score = np.inf # ensure that it be set to 0 from first evaluation :)

    def train(self, verbose=True):
        initial_epoch = self.model.global_step_tensor.eval(self.sess)
        if initial_epoch >= self.config.num_epochs:
            print('model already trained for {} epochs (>= {})'.format(initial_epoch, self.config.num_epochs))
            return 0
        t0 = timer()
        t_eval = 0
        for cur_epoch in range(initial_epoch, self.config.num_epochs):
            result = self.train_epoch()
            summary_dict = []
            if (cur_epoch + 1) % int(self.config.freq_eval) == 0:
                t_eval, summary_dict = self.eval_step(verbose)
                if verbose:
                    print('train epoch result:', result)
                if ('adapt_ranges' in self.config
                        and self.config.adapt_ranges
                        and cur_epoch >= self.config.start_adapt_frac
                        and cur_epoch < self.config.num_epochs * self.config.stop_adapt_frac):
                    self.update_ranges(margin=self.config.margin)
                if self.config.early_stop and self.early_stop(self.config.persistence):
                    print("early stopping at epoch {} with metric {}".format(cur_epoch, self.best_score))
                    break
            self.sess.run(self.model.increment_cur_epoch_tensor)
        tf = timer()
        print('total training time: {} \nAVG time per epoch: {}'
              .format(tf-t0, (tf-t0)/(self.config.num_epochs-initial_epoch)))
        print('including {}s for {} evaluation steps in total'.format(t_eval,
                                                                      self.config.num_epochs
                                                                      // self.config.freq_eval))
        return summary_dict

    def early_stop(self, persistence=5, burn=5, last_val=3):
        if self.accum_early >= persistence:
           #np.mean(self.early_stop_metric_list[-last_val:]) >= np.mean(self.early_stop_metric_list[-persistence:-last_val])):
            #print("{} > {}".format(np.mean(self.early_stop_metric_list[-last_val:]), np.mean(self.early_stop_metric_list[-persistence:-last_val])))
            return True
        else:
            return False



    def train_epoch(self):
        for iteration in range(self.config.num_iter):
            self.train_step()
        cur_it = self.model.global_step_tensor.eval(self.sess)
        self.logger.summarize(cur_it, summaries_dict={})
        self.model.save(self.sess)




