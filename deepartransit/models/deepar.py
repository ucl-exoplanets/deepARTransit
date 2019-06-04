import tensorflow as tf
from .base import BaseModel, BaseTrainer

"""
config file

num_features
num_cov
batch_size
num_timesteps
encoder_length / cond_length
decoder_length / pred_length
learning_rate
num_traces
num_layers
cell_type
cell_args
num_iter
"""
class DeepARModel(BaseModel):
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
            state_at_layer.append(rnn_at_layer[-1].zero_state(self.config.batch_size, dtype=tf.float32))

        loc_decoder = tf.layers.Dense(1)
        scale_decoder = tf.layers.Dense(1, activation='sigmoid')
        loss = tf.Variable(0., dtype=tf.float32, name='loss')

        for t in range(self.config.cond_length):
            if t == 0:  # initialization of input z_0 with zero
                z_prev = tf.zeros(shape=(self.config.batch_size, self.config.num_features))
            elif t and t < self.config.cond_length:  # Conditional range: the input is now simply the previous target z_(t-1)
                z_prev = self.Z[:, t - 1]
            else: # Prediction range (still used for training but with drawn samples)
                sample_z = tf.distributions.Normal(loc, scale).sample(self.config.num_traces)
                self.sample_at_time.append(sample_z)
                z_prev = sample_z

            temp_input = tf.concat([z_prev, self.X[:, t]], axis=-1)
            for layer in range(self.config.num_layers):
                temp_input, state_at_layer[layer] = rnn_at_layer[layer](temp_input, state_at_layer[layer])

            loc = loc_decoder(temp_input)
            scale = scale_decoder(temp_input)

            likelihood = super().gaussian_likelihood(scale)(self.Z[:, t], loc)
            loss = tf.add(loss, likelihood)

            self.loc_at_time.append(loc)
            self.scale_at_time.append(scale)

        with tf.name_scope("loss"):
            self.loss = tf.divide(loss, (self.config.cond_length + self.config.pred_length))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
            self.training_op = self.optimizer.minimize(loss)


class DeepARTrainer(BaseTrainer):

    def __init__(self, sess, model, data, config, logger):
        super().__init__(self, sess, model, data, config, logger)
    def train_epoch(self):
        for iteration in self.config.num_iter:
            batch_Z, batch_X = self.data.next_batch(self.config.batch_size)

            feed_dict = {self.Z: batch_Z, self.X: batch_X}
            self.sess.run([self.model.train_step, self.model.loss], feed_dict=feed_dict)