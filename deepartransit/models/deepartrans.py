import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .base import BaseModel, BaseTrainer

"""
Variant from deepAR original network, adapted to transit light curve structure
"""

class DeepARTransModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        print(self.config)
        self.build()
        self._init_learning_rate()
        super().init_saver()

    def build(self):
        # inputs
        self.Z = tf.placeholder(shape=(None, None, self.config.num_features), dtype=tf.float32, name='instr')
        self.X = tf.placeholder(shape=(None, None, self.config.num_cov), dtype=tf.float32, name='covariates')
        self.alpha = tf.placeholder(shape=(self.config.batch_size,), name='scaled_center', dtype=tf.float32)

        # Transit
        self.trans_pars = tf.Variable(initial_value=self.config.initial_trans_pars, dtype=tf.float32,
                                      name='transit_params', trainable=True)

        self.time_array = tf.Variable(np.linspace(0., 1., self.config.trans_length),
                             name='transit', dtype=tf.float32, trainable=False)

        self.Y = self.transit_linear_tf(self.time_array, self.trans_pars)



        self.pred_at_time = []
        self.scale_at_time = []
        self.sample_at_time = []
        rnn_at_layer = []
        state_at_layer = []
        for _ in range(self.config.num_layers):
            rnn_at_layer.append(tf.nn.rnn_cell.LSTMCell(self.config.hidden_units, **self.config.cell_args))
            #state_at_layer.append(rnn_at_layer[-1].get_initial_state(batch_size=self.config.batch_size, dtype=tf.float32)) # keras version
            state_at_layer.append(rnn_at_layer[-1].zero_state(batch_size=self.config.batch_size, dtype=tf.float32)) # tf.1 version

        loc_decoder = tf.layers.Dense(1)
        scale_decoder = tf.layers.Dense(1, activation='sigmoid')
        loss = tf.Variable(0., dtype=tf.float32, name='loss')
        loss_out = tf.Variable(0., dtype=tf.float32, name='loss_out')
        loss_trans = tf.Variable(0., dtype=tf.float32, name='loss_trans')

        for t in range(self.config.pretrans_length + self.config.trans_length + self.config.postrans_length):
            # initialization of input z_0 with zero
            if t == 0:
                z_prev = tf.zeros(shape=(self.config.batch_size, self.config.num_features))
            elif t < self.config.pretrans_length or t > (self.config.pretrans_length + self.config.trans_length):
                z_prev = self.Z[:, t - 1]
            else: # sample is drawn for whole transit range + first post_transit time ( so trans_length + 1 times)
                sample_z = tfp.distributions.Normal(loc, scale).sample()
                self.sample_at_time.append(sample_z)
                z_prev = sample_z

            if self.config.num_cov:
                temp_input = tf.concat([z_prev, self.X[:, t]], axis=-1)
            else:
                temp_input = z_prev
            for layer in range(self.config.num_layers):
                temp_input, state_at_layer[layer] = rnn_at_layer[layer](temp_input, state_at_layer[layer])

            loc = loc_decoder(temp_input)
            scale = scale_decoder(temp_input)

            if t < self.config.pretrans_length or (t >= self.config.pretrans_length +
                                                   self.config.trans_length):
                z_hat = loc
                likelihood = super().gaussian_likelihood(scale)(self.Z[:, t], z_hat)
                loss_out = tf.add(loss_out, likelihood)
            else:
                delta_t = self.Y[t - self.config.pretrans_length]
                z_hat = tf.add(tf.scalar_mul(delta_t, loc), tf.scalar_mul((delta_t - 1), self.alpha))
                likelihood = super().gaussian_likelihood(scale)(self.Z[:, t], z_hat)
                loss_trans = tf.add(loss_trans, likelihood)
            loss = tf.add(loss, likelihood)

            self.pred_at_time.append(z_hat)
            self.scale_at_time.append(scale)

        with tf.name_scope("loss"):
            self.loss_out = tf.math.divide(loss_out, (self.config.pretrans_length + self.config.trans_length + self.config.postrans_length))
            self.loss_trans = tf.math.divide(loss_trans, (self.config.pretrans_length + self.config.trans_length + self.config.postrans_length))
            self.loss = tf.math.divide(loss, (self.config.pretrans_length + self.config.trans_length + self.config.postrans_length))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
            self.train_step = self.optimizer.minimize(loss, global_step=self.global_step_tensor)
            self.train_step_out = self.optimizer.minimize(loss_out, global_step=self.global_step_tensor)


    @staticmethod
    def transit_linear_tf(time_array, input_pars):
        '''sum of 4 relu'''
        t_c = input_pars[0:1]
        delta = input_pars[1:2]
        T = input_pars[2:3]
        tau = input_pars[3:]

        x1 = time_array - (t_c - T / 2 - tau / 2)
        x2 = time_array - (t_c - T / 2 + tau / 2)
        x3 = time_array - (t_c + T / 2 - tau / 2)
        x4 = time_array - (t_c + T / 2 + tau / 2)
        # star_level = tf.ones(time_array.shape)
        transit_values = 1. - delta / tau * (
                    tf.maximum(x1, 0.) - tf.maximum(x2, 0.) - tf.maximum(x3, 0.) + tf.maximum(x4, 0.))
        return transit_values

class DeepARTransTrainer(BaseTrainer):
    def __init__(self, sess, model, data, config, logger=None):
        super().__init__(sess, model, data, config, logger=logger)

    def train_epoch(self, trans=False):
        batch_Z, batch_X = next(self.data.next_batch(self.config.batch_size))
        feed_dict = {self.model.Z: batch_Z, self.model.X: batch_X,
                     self.model.alpha: (self.data.scaler_Z.centers / self.data.scaler_Z.norms).squeeze((1,2))}

        if trans:
            train_op = self.model.train_step
        else:
            train_op = self.model.train_step_out
        _, loss, loss_out, loss_trans, trans_pars = self.sess.run([train_op, self.model.loss, self.model.loss_out, self.model.loss_trans,
                                                                   self.model.trans_pars], feed_dict= feed_dict)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
                'loss': loss,
                'loss_out': loss_out,
                'loss_trans': loss_trans,
                't_c': trans_pars[0],
                'delta': trans_pars[1],
                'T':trans_pars[2],
                'tau':trans_pars[3],
        }

        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        if cur_it > int(0.5 * self.config.num_epochs) and loss < self.model.best_loss_tensor.eval(self.sess):
            self.sess.run(tf.assign(self.model.best_loss_tensor, tf.constant(loss, dtype='float32')))
            self.model.save(self.sess)
        return loss, loss_out, loss_trans

    def train(self, verbose=False):
        curr_epoch = self.model.cur_epoch_tensor.eval(self.sess)
        if curr_epoch >= self.config.num_epochs:
            print('model already trained for {} epochs (>= {})'.format(curr_epoch, self.config.num_epochs))
            return 0
        for cur_epoch in range(curr_epoch, self.config.num_epochs):
            if verbose:
                print('curr epoch : {} (global step: {})'.format(cur_epoch,
                                                                 self.model.global_step_tensor.eval(self.sess)))
            if cur_epoch <= self.config.num_epochs_out:
                result = self.train_epoch()
            else:
                result = self.train_epoch(trans=True)
                if verbose:
                    print('training with transit')
            self.sess.run(self.model.increment_cur_epoch_tensor)
            if verbose:
                print('train epoch result:', result)

    def sample_sys_traces(self):
        samples_cond_test = np.zeros(shape=(self.config.num_traces, self.config.batch_size,
                                            self.config.trans_length + 1, self.config.num_features))
        for trace in range(self.config.num_traces):
            samples_cond_test[trace] = np.array(
                self.sess.run(self.model.sample_at_time, feed_dict={self.model.Z: self.data.Z,
                                                                    self.model.X: self.data.X})).swapaxes(0, 1)
        return samples_cond_test
