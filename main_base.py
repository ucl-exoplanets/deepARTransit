import tensorflow as tf

from deepartransit.data_handling.data_generator import DataGenerator
from deepartransit.models.base import BaseModel, BaseTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.argumenting import get_args


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)

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
    model.load(sess)
    # here you train your model
    #trainer.train()


if __name__ == '__main__':
    main()