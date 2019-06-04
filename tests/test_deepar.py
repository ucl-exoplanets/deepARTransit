import os
import tensorflow as tf
import yaml
from deepartransit.models import deepar
from deepartransit.utils.config import process_config


config_file = os.path.join('tests', 'deepar_config_test.yml')

def test_deepar_init():
    config = process_config(config_file)
    deepar_model = deepar.DeepARModel(config)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print(os.getcwd(), deepar_model.config.checkpoint_dir)
        deepar_model.save(sess)

'''
def test_deepar_loading():
    config = process_config(config_file)
    deepar_model = deepar.DeepARModel(config)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        deepar_model.load(sess)
'''
