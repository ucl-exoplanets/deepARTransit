import os
import tensorflow as tf
import yaml
from deepartransit.models import deepar
from deepartransit.utils.config import process_config
from deepartransit.data import dataset

config_path = os.path.join('tests', 'deepar_config_test.yml')


def test_deepar_init():
    config = process_config(config_path)
    model = deepar.DeepARModel(config)
    #model.delete_checkpoints()
    data = dataset.DataGenerator(config)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        trainer = deepar.DeepARTrainer(sess, model, data, config)
        trainer.train()

"""
def test_deepar_load():
    config = process_config(config_path)
    model = deepar.DeepARModel(config)
    data = dataset.DataGenerator(config)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        trainer = deepar.DeepARTrainer(sess, model, data, config)
        model.load(sess)
        model.train()
"""