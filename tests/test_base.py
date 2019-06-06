import os
from bunch import Bunch
import tensorflow as tf
from deepartransit.models import base
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger


config = {
    'checkpoint_dir': os.path.join('deepartransit', 'experiments', 'base_test',  'models_checkpoint'),
    'summary_dir': os.path.join('deepartransit', 'experiments', 'base_test', 'summary')
}


def test_base_model():
    base_config = Bunch(config)
    base_model = base.BaseModel(base_config)
    base_model.init_saver()
    base.BaseModel.gaussian_likelihood(1.)
    create_dirs([base_config.summary_dir, base_config.checkpoint_dir])

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        logger = Logger(sess, base_config)
        logger.summarize(0, summaries_dict={})