import os
from deepartransit.utils.config import process_config
from deepartransit.data_handling import data_generator

config_path = os.path.join('tests', 'deepar_config_test.yml')


def test_data():
    config = process_config(config_path)
    data = data_generator.DataGenerator(config)

    batch_Z, batch_X = next(data.next_batch(config.batch_size))
    assert batch_Z.shape[0] == config.batch_size == batch_X.shape[0]
    assert batch_Z.shape[1] == config.cond_length + config.pred_length == batch_X.shape[1]

    Z_test, X_test = data.get_test_data()
    assert Z_test.shape[1] == X_test.shape[1] == config.test_length + config.cond_length



config_path_2 = os.path.join('tests', 'deepar_config_test_2.yml')

def test_data_config_update():
    config = process_config(config_path_2)
    data = data_generator.DataGenerator(config)

    config = data.update_config()
    assert 'num_cov' in config
    assert 'num_features' in config
    assert 'num_ts' in config
    assert config.batch_size == config.num_ts

config_path_3 = os.path.join('tests', 'deeparsys_config_test_2.yml')

def test_data():
    config = process_config(config_path)
    data = data_generator.DataGenerator(config)

    batch_Z, batch_X = next(data.next_batch(config.batch_size))
    assert batch_Z.shape[0] == config.batch_size == batch_X.shape[0]
    assert batch_Z.shape[1] == config.cond_length + config.pred_length == batch_X.shape[1]

    assert data.Z.shape[0] == data.X.shape[0]
    #Z_test, X_test = data.get_test_data()
    #assert Z_test.shape[1] == X_test.shape[1] == config.test_length + config.cond_length
