"""Run various deeparsys models whose config raanges are specified in config file."""
import os

import numpy as np
import tensorflow as tf

from deepartransit.models import deeparsys
from deepartransit.utils import data_generator
from deepartransit.utils.argumenting import get_args
from deepartransit.utils.config import get_config_file, process_config, split_grid_config
from deepartransit.utils.dirs import create_dirs, delete_dirs
from deepartransit.utils.logger import Logger

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import pandas as pd
from timeit import default_timer as timer

if __name__ == '__main__':
    # config_path = os.path.join('deepartransit','experiments', 'deeparsys_dev','deeparsys_config.yml')
    try:
        args = get_args()
        print(args.experiment)
        if args.experiment:
            print('found an experiment argument:', args.experiment)
            meta_config_file = get_config_file(os.path.join("experiments", args.experiment))
            print("which constains a config file", meta_config_file)
        else:
            meta_config_file = args.config
        print('processing the config from the config file')
        meta_config = process_config(meta_config_file)

    except:
        print("missing or invalid arguments")
        exit(0)
    grid_config = split_grid_config(meta_config)
    list_configs = [c for c in grid_config
                    if c['total_length'] == c['pretrans_length'] + c['trans_length'] + c['postrans_length']]

    df_scores = pd.DataFrame(index=list(range(len(list_configs))),
                             columns=list(list_configs[0].keys()) + ['loss_pred', 'nb_epochs', 'mse_pred', 'init_time',
                                                                     'training_time'])
    print('Starting to run {} models'.format(len(list_configs)))
    for i, config in enumerate(list_configs):
        df_scores.loc[i, config.keys()] = list(config.values())
        print('\n\t\t >>>>>>>>> model ', i)
        # print(config)
        data = data_generator.DataGenerator(config)
        config = data.update_config()
        tf.reset_default_graph()

        model = deeparsys.DeepARSysModel(config)

        # model.delete_checkpoints()
        if i:
            delete_dirs([config.summary_dir, config.checkpoint_dir, config.plots_dir])
        else:
            delete_dirs([config.summary_dir, config.checkpoint_dir, config.plots_dir, config.output_dir])
        create_dirs([config.summary_dir, config.checkpoint_dir, config.plots_dir, config.output_dir])

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            t0 = timer()
            sess.run(init)
            if not config.from_scratch:
                model.load(sess)
            logger = Logger(sess, config)

            t1 = timer()
            print('init model time:', t1 - t0)
            trainer = deeparsys.DeepARSysTrainer(sess, model, data, config, logger)
            t2 = timer()
            print('init trainer time:', t2 - t1)

            summary_dict = trainer.train(verbose=True)
            t3 = timer()
            print('training time:', t3 - t2)
            nb_epochs = sess.run(model.cur_epoch_tensor)
            model.load(sess)  # loading best model
            trainer = deeparsys.DeepARSysTrainer(sess, model, data, config, logger)
            samples = trainer.sample_sys_traces()

        print(nb_epochs, summary_dict)
        # print('best_score', trainer.best_score)
        df_scores.loc[i, 'loss_pred'] = summary_dict['loss_pred']
        df_scores.loc[i, 'nb_epochs'] = nb_epochs
        df_scores.loc[i, 'mse_pred'] = summary_dict['mse_pred']
        df_scores.loc[i, 'init_time'] = t2 - t0
        df_scores.loc[i, 'training_time'] = t3 - t2

        df_scores.to_csv(os.path.join("experiments", config.exp_name, 'config_scores.csv'))
        # Saving output array

        np.save(os.path.join(config.output_dir, 'pred_array_{}.npy'.format(i)), np.array(samples))
        print('prediction sample of shape {} saved'.format(np.array(samples).shape))
        print('\n\t\t <<<<<<<< model {} finished'.format(i))
