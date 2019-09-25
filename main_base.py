import os
import tensorflow as tf

from deepartransit.models.base import BaseModel, BaseTrainer
from deepartransit.utils.config import get_config_file, process_config
from deepartransit.utils.dirs import create_dirs
from deepartransit.utils.logger import Logger
from deepartransit.utils.argumenting import get_args


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        print(args.experiment)
        if args.experiment:
            config_file = get_config_file(os.path.join("deepartransit", "experiments", args.experiment))
        else:
            config_file = args.config
        config = process_config(config_file)

    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # create tensorflow session
    sess = tf.Session()
    # create your data_handling generator
    #data = DataGenerator(config)

    # create an instance of the model you want
    model = BaseModel(config)
    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and pass all the previous components to it
    trainer = BaseTrainer(sess, model, None, config, logger)
    # load model if exists
    #model.load(sess)
    # here you train your model
    #trainer.train()
    sess.close()

if __name__ == '__main__':
    main()