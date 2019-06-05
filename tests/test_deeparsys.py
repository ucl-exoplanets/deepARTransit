import os
import tensorflow as tf
import yaml
from deepartransit.models import deeparsys
from deepartransit.utils.config import process_config
from deepartransit.data import dataset
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

config_path = os.path.join('tests', 'deeparsys_config_test.yml')


def test_deepar_init():
    config = process_config(config_path)
    model = deeparsys.DeepARSysModel(config)
    model.delete_checkpoints()
    data = dataset.DataGenerator(config)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        trainer = deeparsys.DeepARSysTrainer(sess, model, data, config)
        trainer.train_step()

        model.load(sess)

        trainer.train()