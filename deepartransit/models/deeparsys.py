import os
from timeit import default_timer as timer
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .base import BaseModel, BaseTrainer
from utils.transit import LinearTransit, LLDTransit

"""
Variant from deepAR original network, adapted to transit light curve structure
"""
#TODO: reformat for pixels as features
class DeepARSysModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.trans_length = ([self.config.trans_length] * self.config.batch_size
                             if isinstance(self.config.trans_length, int) else self.config.trans_length)
        self.pretrans_length = ([self.config.pretrans_length] * self.config.batch_size
                                if isinstance(self.config.pretrans_length, int) else self.config.pretrans_length)
        self.postrans_length = ([self.config.postrans_length] * self.config.batch_size
                                if isinstance(self.config.postrans_length, int) else self.config.postrans_length)
        self.margin_length = [0] * self.config.batch_size
        self.T = self.config.trans_length + self.config.pretrans_length + self.config.postrans_length
        self.build()
        super().init_saver()


    def build(self):
        self.loc_at_time = []
        self.scale_at_time = []
        self.sample_at_time = []

        self.Z = tf.placeholder(shape=(None, None, self.config.num_features), dtype=tf.float32)
        self.X = tf.placeholder(shape=(None, None, self.config.num_cov), dtype=tf.float32)
        self.weights = tf.placeholder(shape=(None, self.config.num_features), dtype=tf.float32)



        rnn_at_layer = []
        state_at_layer = []
        for l in range(self.config.num_layers):
            rnn_at_layer.append(tf.contrib.rnn.LSTMCell(self.config.hidden_units, **self.config.cell_args))

            #state_at_layer.append(rnn_at_layer[-1].get_initial_state(batch_size=self.config.batch_size, dtype=tf.float32)) # keras version
            state_at_layer.append(rnn_at_layer[-1].zero_state(batch_size=self.config.batch_size, dtype=tf.float32)) # tf.1 version
            if 'dropout' in self.config:
                rnn_at_layer[l] = tf.nn.rnn_cell.DropoutWrapper(rnn_at_layer[l], output_keep_prob=1 - self.config.dropout)
                print('using dropout rate {}'.format(self.config.dropout))

        loc_decoder_for = tf.layers.Dense(self.config.num_features)
        scale_decoder_for = tf.layers.Dense(self.config.num_features, activation='sigmoid')

        in_outside_range = lambda t, obs: t < self.pretrans_length[obs] or t >= self.pretrans_length[obs] + \
                                          self.trans_length[obs]
        in_margin_range = lambda t, obs: ((t < self.pretrans_length[obs] + self.margin_length[obs])
                                          or (t >= self.pretrans_length[obs] + self.trans_length[obs] -
                                              self.margin_length[obs]))

        if self.config.bidirectional:
            rnn_back_at_layer = []
            state_back_at_layer = []
            for _ in range(self.config.num_layers):
                rnn_back_at_layer.append(tf.nn.rnn_cell.LSTMCell(self.config.hidden_units, **self.config.cell_args))
                # state_at_layer.append(rnn_at_layer[-1].get_initial_state(batch_size=self.config.batch_size, dtype=tf.float32)) # keras version
                state_back_at_layer.append(
                    rnn_back_at_layer[-1].zero_state(batch_size=self.config.batch_size, dtype=tf.float32))  # tf.1 version

            loc_decoder_back = tf.layers.Dense(self.config.num_features)
            scale_decoder_back = tf.layers.Dense(self.config.num_features, activation='sigmoid')
            loc_decoder = tf.layers.Dense(self.config.num_features)
            scale_decoder = tf.layers.Dense(self.config.num_features, activation='sigmoid')


        loss = tf.Variable(0., dtype=tf.float32, name='loss')
        loss_pred = tf.Variable(0., dtype=tf.float32, name='loss_pred')

        for t in range(self.T):
            # initialization of input z_0 with zero
            if t == 0:
                z_prev = tf.zeros(shape=(self.config.batch_size, self.config.num_features))
            else:
                sample_z = tfp.distributions.Normal(loc, scale).sample()
                self.sample_at_time.append(sample_z)
                z_prev_list = []
                for obs in range(self.config.batch_size):
                    if t < self.pretrans_length[obs] or t > (self.pretrans_length[obs] + self.trans_length[obs]):
                        z_prev_list.append(tf.expand_dims(self.Z[obs, t - 1], 0))
                    else: # sample is drawn for whole transit range + first post_transit time ( so trans_length + 1 times)
                        z_prev_list.append(tf.expand_dims(sample_z[obs],0))

                z_prev = tf.concat(z_prev_list, 0)
                if self.config.add_noise:
                    z_prev = tf.random.normal(shape=z_prev.shape, mean=z_prev, stddev= self.config.noise_level)

            if self.config.num_cov:
                temp_input = tf.concat([z_prev, self.X[:, t]], axis=-1)
            else:
                temp_input = z_prev
            for layer in range(self.config.num_layers):
                temp_input, state_at_layer[layer] = rnn_at_layer[layer](temp_input, state_at_layer[layer])

            loc = loc_decoder_for(temp_input)
            scale = scale_decoder_for(temp_input)
            self.loc_at_time.append(loc)
            self.scale_at_time.append(scale)

        if self.config.bidirectional:
            for t in range(self.T):
                # initialization of input z_0 with zero
                if t == 0:
                    z_prev = tf.zeros(shape=(self.config.batch_size, self.config.num_features))
                else:
                    sample_z = tfp.distributions.Normal(loc, scale).sample()
                    self.sample_at_time.append(sample_z)
                    z_prev_list = []
                    for obs in range(self.config.batch_size):
                        if t < self.postrans_length[obs] or t > (self.postrans_length[obs] + self.trans_length[obs]):
                            z_prev_list.append(tf.expand_dims(self.Z[obs, self.T - t], 0))
                        else:  # sample is drawn for whole transit range + first post_transit time ( so trans_length + 1 times)
                            z_prev_list.append(tf.expand_dims(sample_z[obs],0))
                    z_prev = tf.concat(z_prev_list, 0)
                if self.config.num_cov:
                    temp_input = tf.concat([z_prev, self.X[:, self.T-t-1]], axis=-1)
                else:
                    temp_input = z_prev
                for layer in range(self.config.num_layers):
                    temp_input, state_back_at_layer[layer] = rnn_back_at_layer[layer](temp_input, state_back_at_layer[layer])

                loc = loc_decoder_back(temp_input)
                scale = scale_decoder_back(temp_input)

                self.loc_at_time[self.T - t - 1] = loc_decoder(tf.concat( [self.loc_at_time[self.T - t - 1], loc], -1))
                self.scale_at_time[self.T - t - 1] = scale_decoder(tf.concat( [self.scale_at_time[self.T - t - 1], scale], -1))


        for t in range(self.T):
            for obs in range(self.config.batch_size):
                likelihood = super().gaussian_likelihood(self.scale_at_time[t][obs], self.weights[obs])(self.Z[obs, t],
                                                                                                        self.loc_at_time[
                                                                                                            t][obs])
                if in_outside_range(t, obs) or (self.config.train_margin and in_margin_range(t, obs)):
                    loss = tf.add(loss, tf.math.divide(likelihood, (self.pretrans_length[obs] + self.postrans_length[obs] -
                                                                    (self.margin_length[obs] if self.config.train_margin else 0))))
                else:
                    loss_pred = tf.add(loss_pred, tf.math.divide(likelihood, (self.trans_length[obs] +
                                                                    (self.margin_length[obs] if self.config.train_margin else 0))))
        with tf.name_scope("loss"):
            self.loss = loss
            self.loss_pred = loss_pred
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_tensor)
            self.train_step = self.optimizer.minimize(loss, global_step=self.global_step_tensor)


class DeepARSysTrainer(BaseTrainer):
    def __init__(self, sess, model, data, config, logger=None, transit_model = LinearTransit):
        super().__init__(sess, model, data, config, logger=logger)
        self.transit = [[]] * self.data.Z.shape[0]

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
        if self.config.early_stop and self.early_stop(self.config.persistence):
            self.model.save(self.sess)
        elif cur_it == self.config.num_epochs - 1:
            self.model.save(self.sess)
            #self.best_score =

        return loss_epoch

    def train_step(self):
        batch_Z, batch_X = next(self.data.next_batch(self.config.batch_size))
        if self.config.num_features > 10:
            weights = self.data.scaler_Z.centers.squeeze(0) / self.data.scaler_Z.centers.mean(axis=-1).squeeze(0)
        else:
            weights = np.ones(batch_Z[:,0,:].shape)
        _, loss = self.sess.run([self.model.train_step, self.model.loss],
                                feed_dict={self.model.Z: batch_Z, self.model.X: batch_X, self.model.weights: weights})
        return loss

    def sample_sys_traces(self):
        samples_cond_test = np.zeros(shape=(self.config.num_traces, self.config.batch_size,
                                            self.model.T-1, self.config.num_features))
        for trace in range(self.config.num_traces):
            samples_cond_test[trace] = np.array(
                self.sess.run(self.model.sample_at_time, feed_dict={self.model.Z: self.data.Z,
                                                                    self.model.X: self.data.X})).swapaxes(0, 1)
        return samples_cond_test

    def eval_step(self, verbose=True):
        cur_it = self.model.global_step_tensor.eval()
        if self.config.num_features > 10:
           weights = self.data.scaler_Z.centers.squeeze(0) / self.data.scaler_Z.centers.mean(axis=-1).squeeze(0)
        else:
           weights = np.ones(self.data.Z[:,0,:].shape)

        feed_dict = {self.model.Z: self.data.Z, self.model.X: self.data.X, self.model.weights: weights}

        t1 = timer()
        locs, scales, loss_pred = self.sess.run([self.model.loc_at_time, self.model.scale_at_time, self.model.loss_pred], feed_dict=feed_dict)
        np.save(os.path.join(self.config.output_dir, 'loc_array_{}.npy'.format(cur_it)),
                np.array(locs))
        np.save(os.path.join(self.config.output_dir, 'scales_array_{}.npy'.format(cur_it)),
                np.array(scales))

        t4 = timer()
        if self.config.early_stop:
            self.early_stop_metric_list.append(loss_pred)
        # compute metrics
        ###############
        print(self.data.Z.shape, np.array(scales).shape)
        pred_range = range(self.config.pretrans_length, self.config.pretrans_length+self.config.trans_length)
        mse_pred = np.sqrt(np.mean(((np.take(self.data.Z, pred_range, axis=1) - np.take(locs, pred_range, axis=0).swapaxes(0,1)))**2))
        # TENSORBOARD eval summary
        lr = self.model.learning_rate_tensor.eval()
        summaries_dict = {
                          'loss_pred': np.array(loss_pred),
                          'mse_pred': np.array(mse_pred),
                          'learning_rate': lr
        }

        self.logger.summarize(cur_it, summarizer='test', summaries_dict=summaries_dict)
        return timer() - t1

    def update_ranges(self, margin = 1.05, verbose=True):
        for obs in range(self.data.Z.shape[0]):
            if np.isnan(self.transit[obs].duration) or np.isinf(self.transit[obs].duration):
                duration = self.model.config['trans_length']
            else:
                duration = self.transit[obs].duration
            self.model.trans_length[obs] = int(np.ceil(duration * margin))
            self.model.pretrans_length[obs] = int(np.floor(self.transit[obs].t_c - self.model.trans_length[obs] // 2))
            self.model.postrans_length[obs] = (self.model.T - (self.model.trans_length[obs] + self.model.pretrans_length[obs]))
            self.model.margin_length[obs] = (self.model.trans_length[obs] - self.transit[obs].duration ) // 2
            if verbose:
                print('Transit length recomputed with margin {}: {}'.format(margin, self.model.trans_length))

