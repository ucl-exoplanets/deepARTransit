import os
from deepartransit.models import base


base_config = {
    'checkpoint_dir': os.path.join('tests', 'models_checkpoint')
}


def test_base_model():
    base_model = base.BaseModel(base_config)
    base_model.init_saver()
    base.BaseModel.gaussian_likelihood(1.)


