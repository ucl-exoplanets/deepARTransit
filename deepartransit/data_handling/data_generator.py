import numpy as np
from bunch import Bunch
from pixlc.scaling import MinMaxScaler, MeanStdScaler

class DataGenerator:
    def __init__(self, config):
        self.config = config
        self.Z = np.load(self.config.data_path)
        if self.config.cov_path:
            if isinstance(self.config.cov_path, list):
                self.X = np.concatenate([np.load(path_) for path_ in self.config.cov_path], -1)
            else:
                self.X = np.load(self.config.cov_path)
            if len(self.X) == 1:
                self.X = np.repeat(self.X, len(self.Z), axis=0)
            self._check_consistency()

        if self.config.rescaling:
            self._scale()

    def _scale(self):
        if 'test_length' in self.config:
            train_range = range(self.Z.shape[1] - self.config.test_length)
        else:
            train_range = range(self.Z.shape[1])
        self.scaler_Z = MeanStdScaler(time_axis=1, train_range=train_range)
        self.scaler_X = MeanStdScaler(time_axis=1, train_range=train_range)
        self.Z = self.scaler_Z.fit_transform(self.Z)
        if self.config.cov_path:
            self.X = self.scaler_X.fit_transform(self.X)

    def next_batch(self, batch_size):
        if batch_size:
            idx = np.random.choice(self.Z.shape[0], batch_size)
        else:
            idx = range(self.Z.shape[0])

        if 'cond_length' in self.config and 'pred_length' in self.config and 'test_length' in self.config:
            start_t = np.random.choice(self.Z.shape[1] - self.config.cond_length - self.config.pred_length - self.config.test_length)
            end_t = start_t + self.config.cond_length + self.config.pred_length
        else:
            start_t = 0
            end_t = start_t + self.config.pretrans_length + self.config.trans_length + self.config.postrans_length
        if self.config.cov_path:
            yield (self.Z[idx, start_t:end_t], self.X[idx, start_t:end_t])
        else:
            yield (self.Z[idx, start_t:end_t], None)


    def get_test_data(self, batch_size=0):
        #if batch_size:
        #    idx = np.random.choice(self.Z.shape[0], batch_size)
        #else:
        #    idx = range(self.Z.shape[0])
        cond_test_range = range(self.Z.shape[1] - self.config.test_length - self.config.cond_length, self.Z.shape[1])
        if self.config.cov_path:
            return (self.Z[:, cond_test_range], self.X[:, cond_test_range])
        else:
            return (self.Z[:, cond_test_range], None)

    def _check_consistency(self):
        print(self.Z.shape, self.X.shape)
        try:
            if self.config.cov_path:
                assert self.Z.shape[1] == self.X.shape[1]
        except AssertionError:
            print('inconsistency between Z and X timesteps')
            self.Z = None
            self.X = None
            return -1

        try:
            assert self.config.num_features == self.Z.shape[-1]
            if self.config.cov_path:
                assert self.config.num_cov == self.X.shape[-1]
        except AssertionError:
            print('inconsistency between data_handling and config dimensions')
            self.Z = None
            self.X = None
            return -1

if __name__ == '__main__':
    config_dict = {'data_path': '../data_handling/plc_22807808.npy',
                   'cov_path': '../data_handling/cent_22807808.npy'}
    config = Bunch(config_dict)

    DG = DataGenerator(config)
    print(DG.next_batch(3))
    #print(DG.Z.shape, DG.X.shape)