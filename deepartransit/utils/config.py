
import yaml
from bunch import Bunch
import os
import tensorflow as tf

def get_config_from_yaml(yaml_file):
    """
    Get the config from a json file
    :param yaml_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(yaml_file, 'r') as config_file:
        config_dict = yaml.load(config_file, Loader=yaml.FullLoader)

    # convert the dictionary to a namespace using bunch lib
    config = Bunch(config_dict)

    return config, config_dict


def process_config(yaml_file):
    config, _ = get_config_from_yaml(yaml_file)
    config.summary_dir = os.path.join("deepartransit", "experiments", config.exp_name, "summary/")
    config.checkpoint_dir = os.path.join("deepartransit", "experiments", config.exp_name, "checkpoint/")
    return config

def create_config_file():
    raise NotImplementedError