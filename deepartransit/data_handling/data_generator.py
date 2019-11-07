import numpy as np
from bunch import Bunch
from deepartransit.utils.scaling import MeanStdScaler

class DataGenerator:
    def __init__(self, config):
        self.config = config
        print('loading data from '+ self.config.data_path)
        if 'pretrans_length' in self.config and 'trans_length' in self.config and 'postrans_length' in self.config:
            self.T = config.pretrans_length+config.trans_length+config.postrans_length
        else:
            self.T = None
        self.Z = np.load(self.config.data_path)[:,:self.T]
        self.X = np.zeros(self.Z.shape)
        self.with_cov = not (self.config.cov_path is None or self.config.cov_path=='None' or 'cov_path' not in self.config)
        if self.with_cov:
            if isinstance(self.config.cov_path, list):
                self.X = np.concatenate([np.load(path_) for path_ in self.config.cov_path], -1)
            else:
                if 'pretrans_length' in self.config and 'trans_length' in self.config and 'postrans_length' in self.config:
                    T = config.pretrans_length + config.trans_length + config.postrans_length
                    self.X = np.load(self.config.cov_path)[:, :T]
                else:
                    self.X = np.load(self.config.cov_path)

            if len(self.X) == 1:
                self.X = np.repeat(self.X, len(self.Z), axis=0)
            self._check_consistency()
        try:
            self.time_array = np.load(self.config.time_path)
        except:
            print("time_path parameter not found in config. Default to 0,1,2....T-1")
            self.time_array = np.arange(self.Z.shape[1])
        if self.config.rescaling:
            self._scale()

    def _scale(self):
        if 'test_length' in self.config:
            train_range = range(self.Z.shape[1] - self.config.test_length)
        else:
            train_range = range(self.Z.shape[1])
        self.scaler_Z = MeanStdScaler(time_axis=1, train_range=train_range)
        if self.with_cov:
            self.scaler_X = MeanStdScaler(time_axis=1, train_range=train_range)
        self.Z = self.scaler_Z.fit_transform(self.Z)
        if self.with_cov:
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
            end_t = start_t + self.T
        if self.config['num_cov']:
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
            if 'num_features' in self.config:
                assert self.config.num_features == self.Z.shape[-1]
            if 'num_cov' in self.config:
                assert self.config.num_cov == self.X.shape[-1]
        except AssertionError as e:
            raise e('inconsistency between data_handling and config dimensions')

    def update_config(self, verbose=True):
        '''
        :return: new modified config file with data-related parameters
        '''
        self.config['num_features'] = self.Z.shape[-1]
        self.config['num_cov'] = self.X.shape[-1]
        self.config['num_ts'] = self.Z.shape[0]
        if 'batch_size' not in self.config:
            self.config['batch_size'] = self.config['num_ts']
            if verbose:
                print('Inferring num_features, num_cov, num_ts, batch_size from the data.')
        else:
            if verbose:
                print('Inferring num_features, num_cov, num_ts from the data.')
        return self.config


if __name__ == '__main__':
    config_dict = {'data_path': '../data_handling/plc_22807808.npy',
                   'cov_path': '../data_handling/cent_22807808.npy'}
    config = Bunch(config_dict)

    DG = DataGenerator(config)
    print(DG.next_batch(3))
    #print(DG.Z.shape, DG.X.shape)