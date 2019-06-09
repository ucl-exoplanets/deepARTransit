import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .base import BaseModel, BaseTrainer

"""
Variant from deepAR original network, adapted to transit light curve structure
"""

class DeepARSysModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        print(self.config)
        self.build()
        super().init_saver()

    def build(self):
        self.loc_at_time = []
        self.scale_at_time = []
        self.sample_at_time = []

        self.Z = tf.placeholder(shape=(None, None, self.config.num_features), dtype=tf.float32)
        self.X = tf.placeholder(shape=(None, None, self.config.num_cov), dtype=tf.float32)

        rnn_at_layer = []
        state_at_layer = []
        for _ in range(self.config.num_layers):
            rnn_at_layer.append(tf.nn.rnn_cell.LSTMCell(self.config.hidden_units, **self.config.cell_args))
            state_at_layer.append(rnn_at_layer[-1].zero_state(batch_size=self.config.batch_size, dtype=tf.float32))

        loc_decoder = tf.layers.Dense(1)
        scale_decoder = tf.layers.Dense(1, activation='sigmoid')
        loss = tf.Variable(0., dtype=tf.float32, name='loss')

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

            temp_input = tf.concat([z_prev, self.X[:, t]], axis=-1)
            for layer in range(self.config.num_layers):
                temp_input, state_at_layer[layer] = rnn_at_layer[layer](temp_input, state_at_layer[layer])

            loc = loc_decoder(temp_input)
            scale = scale_decoder(temp_input)

            if t < self.config.pretrans_length or (t >= self.config.pretrans_length +
                                                   self.config.trans_length):
                likelihood = super().gaussian_likelihood(scale)(self.Z[:, t], loc)
                loss = tf.add(loss, likelihood)

            self.loc_at_time.append(loc)
            self.scale_at_time.append(scale)

        with tf.name_scope("loss"):
            self.loss = tf.math.divide(loss, (self.config.pretrans_length + self.config.postrans_length))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
            self.train_step = self.optimizer.minimize(loss, global_step=self.global_step_tensor)


class DeepARSysTrainer(BaseTrainer):
    def __init__(self, sess, model, data, config, logger=None):
        super().__init__(sess, model, data, config, logger=logger)

    def train_epoch(self):
        losses = []
        for iteration in range(self.config.num_iter):
            loss = self.train_step()
            losses.append(loss)
        loss_epoch = np.mean(losses)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'loss': loss_epoch
        }

        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)
        return loss_epoch

    def train_step(self):
        batch_Z, batch_X = next(self.data.next_batch(self.config.batch_size))
        _, loss = self.sess.run([self.model.train_step, self.model.loss],
                                feed_dict={self.model.Z: batch_Z, self.model.X: batch_X})
        return loss

    def sample_sys_traces(self):
        samples_cond_test = np.zeros(shape=(self.config.num_traces, self.config.batch_size,
                                            self.config.trans_length + 1, self.config.num_features))
        for trace in range(self.config.num_traces):
            samples_cond_test[trace] = np.array(
                self.sess.run(self.model.sample_at_time, feed_dict={self.model.Z: self.data.Z,
                                                                    self.model.X: self.data.X})).swapaxes(0, 1)
        return samples_cond_test
