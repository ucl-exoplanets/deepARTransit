import os
from utils.config import process_config
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
