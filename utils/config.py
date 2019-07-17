
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


def process_config(yaml_file, **args):
    config, _ = get_config_from_yaml(yaml_file)
    for k, v in args.items():
        config[k] = v
    config.summary_dir = os.path.join("deepartransit", "experiments", config.exp_name, "summary/")
    config.checkpoint_dir = os.path.join("deepartransit", "experiments", config.exp_name, "checkpoint/")
    config.plots_dir = os.path.join("deepartransit", "experiments", config.exp_name, "plots/")
    config.output_dir =  os.path.join("deepartransit", "experiments", config.exp_name, "output/")

    if 'stop_adapt_frac' not in config:
        config['stop_adapt_frac'] = 0.5
    if 'bidirectional' not in config:
        config['bidirectional'] = False
    if 'train_margin' not in config:
        config['bidirectional'] = True
    return config

def get_config_file(dir_, file_name=None, extension='.yml'):
    cond_on_name = lambda f: f==file_name if (file_name is not None) else ('config' in f and extension in f)
    candidates = [ f for f in os.listdir(dir_) if cond_on_name(f)]
    try:
        assert len(candidates) == 1
        return os.path.join(dir_, candidates[0])
    except AssertionError:
        if len(candidates):
            print('More than one config file found in dir')
        else:
            print('no config file found in dir')
        exit(0)

