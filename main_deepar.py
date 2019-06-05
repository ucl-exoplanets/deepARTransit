import os
import tensorflow as tf
import yaml
from deepartransit.models import deepar
from deepartransit.utils.config import process_config
from deepartransit.data import dataset


if __name__ == '__main__':

    config_path = os.path.join('tests', 'deepar_config_test.yml')
    config = process_config(config_path)
    model = deepar.DeepARModel(config)
    data = dataset.DataGenerator(config)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        trainer = deepar.DeepARTrainer(sess, model, data, config)

        model.load(sess)
        for i in range(10):
            print(model.global_step_tensor.eval(sess))
            print(model.cur_epoch_tensor.eval(sess))
            loss_epoch = trainer.train_epoch()
            print(loss_epoch)