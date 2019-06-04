import os
import tensorflow as tf
from deepartransit.utils.config import process_config


config_file = os.path.join('tests', 'deepar_config_test.yml')

def test_config():
    config = process_config(config_file)
    print(config.cell_args)
