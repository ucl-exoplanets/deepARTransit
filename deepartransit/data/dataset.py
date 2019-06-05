import numpy as np
from bunch import Bunch

class DataGenerator:
    def __init__(self, config):
        self.config = config
        self.Z = np.load(self.config.data_path)
        self.X = np.load(self.config.cov_path)
        if len(self.X) == 1:
            self.X = np.repeat(self.X, len(self.Z), axis=0)
        self._check_consistency()

    def next_batch(self, batch_size):
        idx = np.random.choice(self.Z.shape[0], batch_size)
        start_t = np.random.choice(self.Z.shape[1] - self.config.cond_length - self.config.pred_length - self.config.test_length)
        end_t = start_t + self.config.cond_length + self.config.pred_length
        yield (self.Z[idx, start_t:end_t], self.X[idx, start_t:end_t])


    def _check_consistency(self):
        """

        :return:
        """
        try:
            assert self.Z.shape[1] == self.X.shape[1]
        except AssertionError:
            print('inconsistency between Z and X timesteps')
            self.Z = None
            self.X = None
            return -1

        try:
            assert self.config.num_features == self.Z.shape[-1]
            assert self.config.num_cov == self.X.shape[-1]
        except AssertionError:
            print('inconsistency between data and config dimensions')
            self.Z = None
            self.X = None
            return -1

if __name__ == '__main__':
    config_dict = {'data_path': '../data/plc_22807808.npy',
                   'cov_path': '../data/cent_22807808.npy'}
    config = Bunch(config_dict)

    DG = DataGenerator(config)
    print(DG.next_batch(3))
    #print(DG.Z.shape, DG.X.shape)