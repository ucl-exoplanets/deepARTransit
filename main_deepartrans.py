import os

import numpy as np
import tensorflow as tf

from deepartransit.models.deepartrans import DeepARTransModel, DeepARTransTrainer
from deepartransit.utils import data_generator
from deepartransit.utils.argumenting import get_args
from deepartransit.utils.config import get_config_file, process_config
from deepartransit.utils.dirs import create_dirs
from deepartransit.utils.logger import Logger
from deepartransit.utils.transit import get_transit_model

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == '__main__':
    # config_path = os.path.join('deepartransit','experiments', 'deeparsys_dev','deeparsys_config.yml')
    try:
        args = get_args()
        print(args.experiment)
        if args.experiment:
            print('found an experiment argument:', args.experiment)
            config_file = get_config_file(os.path.join("experiments", args.experiment))
            print("which constains a config file", config_file)
        else:
            config_file = args.config
        print('processing the config from the config file')
        config = process_config(config_file)

    except:
        print("missing or invalid arguments")
        exit(0)

    tf.reset_default_graph()
    data = data_generator.DataGenerator(config)
    config = data.update_config()
    model = DeepARTransModel(config)

    if config.from_scratch:
        model.delete_checkpoints()

    create_dirs([config.summary_dir, config.checkpoint_dir, config.plots_dir, config.output_dir])

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        if not config.from_scratch:
            model.load(sess)
        logger = Logger(sess, config)

        transit_model = get_transit_model(config['transit_model'])
        print(transit_model)
        trainer = DeepARTransTrainer(sess, model, data, config, logger, transit_model)
        trainer.train(verbose=True)

        print(data.Z.shape, data.X.shape)
        trainer = DeepARTransTrainer(sess, model, data, config, logger, transit_model)
        samples = trainer.sample_sys_traces()

    # Saving output array
    np.save(os.path.join(config.output_dir, 'pred_array.npy'), np.array(samples))
    print('prediction sample of shape {} saved'.format(np.array(samples).shape))
