import os
import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt
from deepartransit.models import deeparsys
from deepartransit.utils.config import process_config
from deepartransit.data import dataset
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


if __name__ == '__main__':
    config_path = os.path.join('deepartransit','experiments', 'deeparsys_dev','deeparsys_config.yml')
    config = process_config(config_path)
    model = deeparsys.DeepARSysModel(config)
    data = dataset.DataGenerator(config)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        trainer = deeparsys.DeepARSysTrainer(sess, model, data, config)
        if trainer.config.from_scratch:
            model.delete_checkpoints()
        model.load(sess)
        trainer.train(add_epochs=trainer.config.add_epochs, verbose=True)
        samples = trainer.sample_sys_traces()
    print(np.array(samples).shape)

    # Look at predictions on 'transit' range
    t1 = model.config.pretrans_length
    t2 = t1 + model.config.trans_length
    t3 = t2 + model.config.postrans_length
    plt.figure()
    for pixel in range(samples.shape[1]):
        plt.plot(data.Z[pixel, :, 0], label='ground truth', color='blue')
        plt.plot(data.X[pixel], color='grey', linewidth=1, linestyle='dashed', label='centroid')
        for trace in range(samples.shape[0]):
            plt.plot(range(t1, t2 +1), samples[trace, pixel, :, 0], alpha=0.3, linewidth=1)
        plt.plot(range(t1, t2 + 1), samples[:, pixel, :, 0].mean(axis=0), linestyle=':', color='red', linewidth=4)
        plt.axvline(config.pretrans_length, 0, 1, linestyle='dashed', color='red')
        plt.axvline(config.pretrans_length + config.trans_length, 0, 1, linestyle='dashed', color='red')
        plt.xlim(0, t3)
        plt.legend()
        plt.show()