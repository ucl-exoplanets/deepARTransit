import os
from utils.config import process_config, get_config_file


config_file = os.path.join('tests', 'deepar_config_test.yml')

def test_config():
    try:
        get_config_file('tests')
    except SystemExit:
        print("it's okay, there are probably several config files")
    config = process_config(config_file)
    print(config.cell_args)
