
import yaml
from bunch import Bunch
import os
#import tensorflow as tf
from sklearn.model_selection import ParameterGrid

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
    config.summary_dir = os.path.join("experiments", config.exp_name, "summary/")
    config.checkpoint_dir = os.path.join("experiments", config.exp_name, "checkpoint/")
    config.plots_dir = os.path.join("experiments", config.exp_name, "plots/")
    config.output_dir =  os.path.join("experiments", config.exp_name, "output/")


    if 'early_stop' not in config:
        config['early_stop'] = False
        print('early_stop defaulted to False')
    if 'bidirectional' not in config:
        config['bidirectional'] = False
        print('defaulting bidirectional to False')
    if 'add_noise' not in config:
        config['add_noise'] = False
        print('add_noise defaulted to False')
    if 'adapt_range' not in config:
        config['adapt_range'] = True
        print('defaulting adapt range to True')
    if config['adapt_range'] and 'stop_adapt_frac' not in config:
        config['stop_adapt_frac'] = 0.5
        print('defaulting stop adapt frac to 0.5')
    if config['adapt_range'] and 'start_adapt_frac' not in config:
        config['start_adapt_frac'] = 0.01
        print('defaulting start adapt frac to 0.5')
    if config['adapt_range'] and 'margin' not in config:
        config['margin'] = 1.03
        print('defaulting stop adapt frac to 0.5')
    if 'train_margin' not in config:
        config['train_margin'] = True
        print('defaulting train_margin to True')
    if 'starter_learning_rate' in config:
        if 'end_learning_rate' not in config:
            config['end_learning_rate'] = config['starter_learning_rate'] / 100
        if 'power' not in config:
            config['power'] = 5
        if 'decay_steps' not in config:
            config['decay_steps'] = config['num_epochs']
    else:
        if 'learning_rate' not in config:
            config['learning_rate'] = 0.001
    if 'transit_model' not in config:
        config['transit_model'] = 'linear'
        print('defaulting transit mode to linear')
    return config


def split_grid_config(config):
    """returns the list of individual config objects, considering the grid product
    of all the collection elements present in the input config object"""
    param_grid = Bunch(dict())
    for k,v in config.items():
        try:
            assert isinstance(v, list) or isinstance(v, tuple)
            param_grid[k] = v
        except AssertionError:
            param_grid[k] = [v]
            continue
    return [Bunch(d) for d in ParameterGrid(param_grid)]


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

