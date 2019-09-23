import os
import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt
from deepartransit.models import deepar
from deepartransit.utils.config import process_config, get_config_file
from deepartransit.utils.dirs import create_dirs
from deepartransit.utils.logger import Logger
from deepartransit.utils.argumenting import get_args
from deepartransit.data_handling import data_generator
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


if __name__ == '__main__':
    #config_path = os.path.join('deepartransit','experiments', 'deepar_dev','deepar_config.yml')

    try:
        args = get_args()
        print(args.experiment)
        if args.experiment:
            print('ok')
            config_file = get_config_file(os.path.join("deepartransit", "experiments", args.experiment.strip()))
            print(config_file)
        else:
            config_file = args.config
        print('ok2')
        config = process_config(config_file)

    except:
        print("missing or invalid arguments")
        exit(0)

    model = deepar.DeepARModel(config)
    data = data_generator.DataGenerator(config)

    create_dirs([config.summary_dir, config.checkpoint_dir, config.plots_dir, config.output_dir])

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        logger = Logger(sess, config)
        trainer = deepar.DeepARTrainer(sess, model, data, config, logger)

        if trainer.config.from_scratch:
            model.delete_checkpoints()
        model.load(sess)

        trainer.train(verbose=True)
        samples = trainer.sample_on_test()

    # Saving output array
    np.save(os.path.join(config.output_dir, 'pred_array.npy'), np.array(samples))
    print('prediction sample of shape {} saved'.format(np.array(samples).shape))


    Z_test, X_test = data.get_test_data()
    plt.figure()
    for pixel in range(samples.shape[1]):
        plt.clf()
        plt.plot(Z_test[pixel])
        for trace in range(samples.shape[0]):
            plt.plot(samples[trace, pixel, :, 0], color='orange')
        plt.axvline(config.cond_length, 0, 1, linestyle='dashed', color='red')
        plt.savefig(os.path.join(model.config.plots_dir, 'pixel{}.png'.format(pixel)))
        #plt.show()


