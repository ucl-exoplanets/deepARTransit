import os
import tensorflow as tf
from deepartransit.models import deeparsys
from utils.config import process_config
from utils.dirs import create_dirs
from deepartransit.data_handling import data_generator
from utils.logger import Logger
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

config_path = os.path.join('tests', 'deeparsys_config_test.yml')


def test_deepar_init():
    config = process_config(config_path)
    create_dirs([config.summary_dir, config.checkpoint_dir, config.plots_dir, config.output_dir])
    assert os.path.exists(config.summary_dir)
    model = deeparsys.DeepARSysModel(config)
    model.delete_checkpoints()
    data = data_generator.DataGenerator(config)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        logger = Logger(sess, config)
        trainer = deeparsys.DeepARSysTrainer(sess, model, data, config, logger)
        trainer.eval_step()
        trainer.train_step()

        model.load(sess)

        trainer.train()
        trainer.eval_step()