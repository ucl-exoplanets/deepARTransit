import os

from deepartransit.utils.config import process_config, get_config_file, split_grid_config

config_file = os.path.join('tests', 'deepar_config_test.yml')


def test_config():
    try:
        get_config_file('tests')
    except SystemExit:
        print("it's okay, there are probably several config files")
    config = process_config(config_file)
    print(config.cell_args)


def test_splitting_config():
    config = process_config(config_file)

    config_list = split_grid_config(config)

    assert isinstance(config_list, list)
    assert len(config_list)
    for c in config_list:
        for k, v in c.items():
            print(k, v)
            assert not isinstance(v, list)
