import os
import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt
from deepartransit.models import deepar
from deepartransit.utils.config import process_config
from deepartransit.data import dataset
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


if __name__ == '__main__':
    config_path = os.path.join('deepartransit','experiments', 'deepar_dev','deepar_config.yml')
    config = process_config(config_path)
    model = deepar.DeepARModel(config)
    data = dataset.DataGenerator(config)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        trainer = deepar.DeepARTrainer(sess, model, data, config)
        model.delete_checkpoints()
        model.load(sess)
        trainer.train(add_epochs=3, verbose=True)

        #Z_test, X_test = data.get_test_data()
        #print(Z_test.shape, X_test.shape)
        #samples = sess.run(model.loc_at_time, feed_dict={model.Z: Z_test, model.X: X_test})
        samples = trainer.sample_on_test()
    print(np.array(samples).shape)



    Z_test, X_test = data.get_test_data()

    for pixel in range(samples.shape[1]):
        plt.plot(Z_test[pixel, :, 0])
        for trace in range(samples.shape[0]):
            plt.plot(samples[trace, pixel, :, 0], color='orange')
        plt.axvline(config.cond_length, 0, 1, linestyle='dashed', color='red')
        plt.show()


