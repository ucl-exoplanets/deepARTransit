import os
import yaml
from deepartransit.utils.config import process_config
from deepartransit.models import deepar
from deepartransit.data import dataset

config_path = os.path.join('tests', 'deepar_config_test.yml')


def test_data():
    config = process_config(config_path)
    data = dataset.DataGenerator(config)

    batch_Z, batch_X = next(data.next_batch(config.batch_size))
    assert batch_Z.shape[0] == config.batch_size == batch_X.shape[0]
    assert batch_Z.shape[1] == config.cond_length + config.pred_length == batch_X.shape[1]
