import os

import tensorflow as tf
from bunch import Bunch

from deepartransit.models import base
from deepartransit.utils.dirs import create_dirs
from deepartransit.utils.logger import Logger

config = {
    'checkpoint_dir': os.path.join('experiments', 'base_test', 'models_checkpoint'),
    'summary_dir': os.path.join('experiments', 'base_test', 'summary')
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
