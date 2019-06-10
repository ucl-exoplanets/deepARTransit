import os
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
    try:
        args = get_args()
        print(args.experiment)
        if args.experiment:
            print('found an experiment argument:', args.experiment)
            config_file = get_config_file(os.path.join("deepartransit", "experiments", args.experiment))
            print("which constains a config file", config_file)
        else:
            config_file = args.config
        print('processing the config from the config file')
        config = process_config(config_file)

    except:
        print("missing or invalid arguments")
        exit(0)

    model = deeparsys.DeepARSysModel(config)
    data = data_generator.DataGenerator(config)

    create_dirs([config.summary_dir, config.checkpoint_dir, config.plots_dir, config.output_dir])

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        logger = Logger(sess, config)
        trainer = deeparsys.DeepARSysTrainer(sess, model, data, config, logger)
        if trainer.config.from_scratch:
            model.delete_checkpoints()
        model.load(sess)
        trainer.train(verbose=True)
        samples = trainer.sample_sys_traces()

    # Saving output array
    np.save(os.path.join(config.output_dir, 'pred_array.npy'), np.array(samples))
    print('prediction sample of shape {} saved'.format(np.array(samples).shape))

    # Look at predictions on 'transit' range
    t1 = model.config.pretrans_length
    t2 = t1 + model.config.trans_length
    t3 = t2 + model.config.postrans_length

    plt.figure()
    for pixel in range(samples.shape[1]):
        plt.clf()
        plt.plot(data.Z[pixel, :, 0], label='ground truth', color='blue')
        plt.plot(data.X[pixel], color='grey', linewidth=1, linestyle='dashed', label='centroid')
        for trace in range(samples.shape[0]):
            plt.plot(range(t1, t2 +1), samples[trace, pixel, :, 0], alpha=0.3, linewidth=1)
        plt.plot(range(t1, t2 + 1), samples[:, pixel, :, 0].mean(axis=0), linestyle=':', color='red', linewidth=4)
        plt.axvline(config.pretrans_length, 0, 1, linestyle='dashed', color='red')
        plt.axvline(config.pretrans_length + config.trans_length, 0, 1, linestyle='dashed', color='red')
        plt.xlim(0, t3)
        plt.legend()
        plt.savefig(os.path.join(model.config.plots_dir, 'pixel{}.png'.format(pixel)))
        #plt.show()