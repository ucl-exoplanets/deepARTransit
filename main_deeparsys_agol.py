import os
import yaml
import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt
from deepartransit.models import deeparsys
from utils.config import get_config_file, process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.argumenting import get_args
from deepartransit.data_handling import data_generator
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


if __name__ == '__main__':
    #config_path = os.path.join('deepartransit','experiments', 'deeparsys_dev','deeparsys_config.yml')
    config_dict = {}
    agol_aorkeys = ['22807296', '22807552', '22807808', '24537856', '27603712', '27773440']
    for aorkey in agol_aorkeys:
        try:
            args = get_args()
            config_file = args.config
            print('processing the config from the config file')

            if 'radius' in args:
                radius = args.radius
            else:
                print('selecting default radius =', 3)
                radius = 3
        except:
            print("missing or invalid arguments")
            exit(0)

        config_dict[aorkey] = process_config(config_file,
                                             exp_name='cobweb/agol_artif_transits_bidirect/{}'.format(aorkey),
                                             data_path = 'deepartransit/data/agol_transits_r{}_nobacksub/rlc_artif_{}.npy'.format(radius, aorkey),
                                             cov_path = 'deepartransit/data/agol_transits_r{}_nobacksub/cent_{}.npy'.format(radius, aorkey))
        print(config_dict[aorkey])
        create_dirs([os.path.join('deepartransit', 'experiments', config_dict[aorkey].exp_name)])
        with open(os.path.join("deepartransit", "experiments", config_dict[aorkey].exp_name, 'config.yml'), 'w') as f:
            yaml.dump({k:config_dict[aorkey][k] for k in config_dict[aorkey].keys()}, f)
        model = deeparsys.DeepARSysModel(config_dict[aorkey])
        create_dirs([os.path.join('deepartransit', 'experiments', config_dict[aorkey].exp_name)])

        data = data_generator.DataGenerator(config_dict[aorkey])

        if config_dict[aorkey].from_scratch:
            model.delete_checkpoints()

        create_dirs([config_dict[aorkey].summary_dir, config_dict[aorkey].checkpoint_dir, config_dict[aorkey].plots_dir, config_dict[aorkey].output_dir])

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)

            model.load(sess)
            logger = Logger(sess, config_dict[aorkey])
            trainer = deeparsys.DeepARSysTrainer(sess, model, data, config_dict[aorkey], logger)
            trainer.train(verbose=True)
