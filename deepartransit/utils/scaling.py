import warnings

import numpy as np


class Scaler():
    def __init__(self, train_range=None, centers=None, norms=None, time_axis=0):
        self.train_range = None
        self.time_axis = time_axis
        self._update_range(train_range)
        self.centers = centers
        self.norms = norms

    def _update_range(self, train_range=None, plc=None):
        if train_range is not None:
            self.train_range = train_range
        elif self.train_range is None and plc is not None:
            self.train_range = range(len(plc))

    def fit(self, plc, train_range=None):
        self._update_range(train_range, plc)

    def transform(self, plc):
        if self.zero_norm:
            raise ZeroDivisionError
        return (plc - self.centers) / self.norms

    def fit_transform(self, plc, train_range=None):
        self._update_range(train_range, plc)
        self.fit(plc)
        return self.transform(plc)

    def inverse_transform(self, plc_s):
        return (plc_s * self.norms) + self.centers

    def check_consistency(self):
        if np.isclose(0., self.norms).any():
            self.zero_norm = True
            warnings.warn('Some time-series have zero norm.', Warning)
        else:
            self.zero_norm = False


class MinMaxScaler(Scaler):
    def __init__(self, train_range=None, time_axis=0):
        super().__init__(train_range=train_range, time_axis=time_axis)

    def fit(self, plc, train_range=None):
        super().fit(plc, train_range)
        self.centers = np.expand_dims(plc.take(self.train_range, self.time_axis).min(axis=self.time_axis),
                                      self.time_axis)
        self.norms = np.expand_dims((plc.take(self.train_range, self.time_axis).max(axis=self.time_axis) - plc.take(
            self.train_range, self.time_axis).min(axis=self.time_axis)), self.time_axis)
        super().check_consistency()


class MeanStdScaler(Scaler):
    def __init__(self, train_range=None, time_axis=0):
        super().__init__(train_range=train_range, time_axis=time_axis)

    def fit(self, plc, train_range=None):
        super().fit(plc, train_range)
        self.centers = np.expand_dims(plc.take(self.train_range, self.time_axis).mean(axis=self.time_axis),
                                      self.time_axis)
        self.norms = np.expand_dims(plc.take(self.train_range, self.time_axis).std(axis=self.time_axis), self.time_axis)
        super().check_consistency()
