import os
import tensorflow as tf
import yaml
from deepartransit.models import deepar
from deepartransit.utils.config import process_config
from deepartransit.data import dataset

config_path = os.path.join('tests', 'deepar_config_test.yml')

def test_deepar_init():
    config = process_config(config_path)
    deepar_model = deepar.DeepARModel(config)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print(os.getcwd(), deepar_model.config.checkpoint_dir)
        deepar_model.save(sess)






def test_deepar_training():
    config = process_config(config_path)

    deepar_model = deepar.DeepARModel(config)

    data = dataset.DataGenerator(config)


    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        deepar_trainer = deepar.DeepARTrainer(sess, deepar_model, data, config)

        deepar_trainer.train_step()
        #deepar_model.load(sess)