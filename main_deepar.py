import os
import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt
from deepartransit.models import deepar
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.argumenting import get_args
from deepartransit.data_handling import data_generator
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


if __name__ == '__main__':
    #config_path = os.path.join('deepartransit','experiments', 'deepar_dev','deepar_config.yml')

    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    model = deepar.DeepARModel(config)
    data = data_generator.DataGenerator(config)

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
    print(np.array(samples).shape)

    Z_test, X_test = data.get_test_data()

    for pixel in range(samples.shape[1]):
        plt.plot(Z_test[pixel, :, 0])
        for trace in range(samples.shape[0]):
            plt.plot(samples[trace, pixel, :, 0], color='orange')
        plt.axvline(config.cond_length, 0, 1, linestyle='dashed', color='red')
        plt.show()


