import warnings

import matplotlib.cbook
import matplotlib.pylab as plt
import numpy as np
import scipy.optimize as opt
from pylightcurve import transit, transit_duration

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)


# TODO: tensorflow implementation

class Transit:
    def __init__(self, time_array, transit_pars=None):
        self.time_array = time_array
        self.transit_pars = transit_pars
        self.popt = None
        self.pcov = None
        # self._default_pars()

    @staticmethod
    def _compute_flux(time_array, *transit_pars):
        raise NotImplementedError

    def get_flux(self, time_array=None, transit_pars=None, whole_range=True):
        if time_array is None:
            time_array = self.time_array
        if not whole_range:
            time_array = time_array[self.range_fit]
        if (transit_pars is None) and (self.transit_pars is not None):
            transit_pars = self.transit_pars
        elif transit_pars is None:
            raise ValueError('no transit pars found.')
        return self._compute_flux(time_array, *transit_pars)

    def _default_pars(self, p0=None, bounds=None):
        raise NotImplementedError

    def fit(self, data, p0=None, bounds=None, range_fit=None, sigma=None, time_axis=1, replace_pars=True):
        if range_fit is None:
            self.range_fit = range(0, self.time_array.shape[-1])
        else:
            self.range_fit = range_fit
            assert isinstance(range_fit, range)

        if data.shape[time_axis] == len(self.time_array):
            self.data_fit = np.take(data, self.range_fit, axis=time_axis).flatten()
        elif data.shape[time_axis] == len(self.range_fit):
            self.data_fit = data.flatten()
        else:
            raise ValueError('data shape {}, range_fit len = {}'.format(data.shape, len(self.range_fit)))
        self._default_pars(p0, bounds)

        popt, pcov = opt.curve_fit(self._compute_flux,
                                   self.time_array[self.range_fit],
                                   self.data_fit,
                                   sigma=sigma,
                                   absolute_sigma=True,
                                   p0=self.p0,
                                   bounds=self.bounds,
                                   maxfev=1_000_000)

        self.popt = popt
        self.pcov = pcov
        if replace_pars:
            self.transit_pars = popt
        return 0

    def plot(self, **plot_args):
        plt.plot(self.time_array, self.flux, **plot_args)

    def _get_duration(self):
        raise NotImplementedError

    def _get_t_c(self):
        raise NotImplementedError

    def _get_delta(self):
        raise NotImplementedError

    def _get_err(self):
        return np.sqrt(np.diag(self.pcov))

    def _get_err_delta(self):
        return NotImplementedError

    flux = property(get_flux)
    duration = property(_get_duration)
    t_c = property(_get_t_c)
    delta = property(_get_delta)
    err = property(_get_err)
    err_delta = property(_get_err_delta)


class LinearTransit(Transit):
    def __init__(self, time_array, transit_pars=None):
        super().__init__(time_array, transit_pars)

    def _default_pars(self, p0=None, bounds=None):
        duration = self.time_array[self.range_fit][-1] - self.time_array[self.range_fit[0]]
        if p0 is None:
            self.p0 = (np.median(self.time_array[self.range_fit]), 0.01, duration / 2, duration / 10)
        else:
            self.p0 = p0
        if bounds is None:
            self.bounds = ((self.time_array[self.range_fit][0],
                            0.,
                            duration / len(self.range_fit),
                            duration / len(self.range_fit)),
                           (self.time_array[self.range_fit][-1],
                            0.5,
                            duration,
                            duration / 5))
        else:
            self.bounds = bounds

    def __repr__(self):
        s = "Piecewise linear transit model "
        if self.transit_pars is None:
            s += "(t_c ; delta ; T ; tau)"
        else:
            s += "(t_c = {:.3g} ; delta = {:.3g} ; T = {:.3g} ; tau = {:.3g})".format(*self.transit_pars)
        return s

    @staticmethod
    def _compute_flux(time_array, t_c, delta, T, tau):
        x1 = time_array - (t_c - T / 2 - tau)
        x2 = time_array - (t_c - T / 2)
        x3 = time_array - (t_c + T / 2)
        x4 = time_array - (t_c + T / 2 + tau)
        return 1. - delta / tau * (np.maximum(x1, 0.) - np.maximum(x2, 0.) - np.maximum(x3, 0.) + np.maximum(x4, 0.))

    def _get_duration(self):
        T, tau = self.transit_pars[-2:]
        return (T + 2 * tau)

    def _get_t_c(self):
        return self.transit_pars[0]

    def _get_delta(self):
        return self.transit_pars[1]

    def _get_err_delta(self):
        return self.err[1]

    delta = property(_get_delta)
    duration = property(_get_duration)
    t_c = property(_get_t_c)
    err_delta = property(_get_err_delta)


class LLDTransit(Transit):
    def __init__(self, time_array, transit_pars=None):
        super().__init__(time_array, transit_pars)

    def __repr__(self):
        s = "Linear Limb Darkening transit model "
        if self.transit_pars is None:
            s += "(u, rp_over_rs, period, sma_over_rs, inclination, eccentricity, periastron, mid_time)"
        else:
            s += "(u = {} ; rp_over_rs = {:.3g} ; period = {:.3g} ; sma_over_rs = {:.3g} ; inclination = {:.3g} ; "
            s += "eccentricity = {:.3g} ; periastron = {:.3g} ; mid_time= {:.3g} )"
            s = s.format(*self.transit_pars)
        return s

    def _default_pars(self, p0=None, bounds=None):
        duration = self.time_array[self.range_fit][-1] - self.time_array[self.range_fit[0]]

        if p0 is None:
            self.p0 = [0.05,
                       0.01,
                       duration * 2,
                       1,
                       88,
                       0.05,
                       0,  # periastron
                       np.median(self.time_array[self.range_fit])
                       ]

        else:
            self.p0 = p0
        if bounds is None:
            self.bounds = [(0.,
                            0.,
                            duration / 2,
                            1.,
                            45.,  # inclination
                            0.,  # eccentricity
                            0.,  # periastron
                            self.time_array[self.range_fit][0]),
                           (1.5,
                            0.5,
                            duration * 10,
                            100_000,
                            90.,
                            0.5,
                            90.,  # periastron
                            self.time_array[self.range_fit][-1]
                            )]
        else:
            self.bounds = bounds

    @staticmethod
    def _compute_flux(time_array, u, rp_over_rs, period, sma_over_rs, inclination, eccentricity, periastron, mid_time):

        return transit('linear', [u], rp_over_rs, period, sma_over_rs, eccentricity, inclination,
                       periastron=0., mid_time=mid_time, time_array=time_array, precision=6)

    def _get_duration(self):
        return transit_duration(*self.transit_pars[1:7])

    def _get_delta(self):
        return self.transit_pars[1] ** 2

    def _get_err_delta(self):
        return self.err[1] ** 2

    def _get_t_c(self):
        return self.transit_pars[-1]

    delta = property(_get_delta)
    err_delta = property(_get_err_delta)
    duration = property(_get_duration)
    t_c = property(_get_t_c)


class QLDTransit(Transit):
    def __init__(self, time_array, transit_pars=None):
        super().__init__(time_array, transit_pars)

    def _default_pars(self, p0=None, bounds=None):
        duration = self.time_array[self.range_fit][-1] - self.time_array[self.range_fit[0]]

        if p0 is None:
            self.p0 = [0.05,
                       0.,
                       0.,
                       0.,
                       0.01,
                       duration * 2,
                       1,
                       88,
                       0.05,
                       0,  # periastron
                       np.median(self.time_array[self.range_fit])
                       ]

        else:
            self.p0 = p0
        if bounds is None:
            self.bounds = [(0.,
                            -1.,
                            -1.,
                            -1.,
                            0.,
                            duration / 2,
                            1.,
                            45.,  # inclination
                            0.,  # eccentricity
                            0.,  # periastron
                            self.time_array[self.range_fit][0]),
                           (1.5,
                            2.,
                            2.,
                            2.,
                            0.5,
                            duration * 10,
                            100_000,
                            90.,
                            0.5,
                            90.,  # periastron
                            self.time_array[self.range_fit][-1]
                            )]
        else:
            self.bounds = bounds

    @staticmethod
    def _compute_flux(time_array, ldc1, ldc2, ldc3, ldc4, rp_over_rs, period, sma_over_rs,
                      inclination, eccentricity, periastron, mid_time):

        return transit('claret', [ldc1, ldc2, ldc3, ldc4], rp_over_rs, period, sma_over_rs, eccentricity, inclination,
                       periastron=0., mid_time=mid_time, time_array=time_array, precision=6)

    def _get_duration(self):
        return transit_duration(*self.transit_pars[4:10])

    def _get_delta(self):
        return self.transit_pars[4] ** 2

    def _get_err_delta(self):
        return self.err[4] ** 2

    def _get_t_c(self):
        return self.transit_pars[-1]

    delta = property(_get_delta)
    err_delta = property(_get_err_delta)
    duration = property(_get_duration)
    t_c = property(_get_t_c)


def get_transit_model(model='linear'):
    if model.lower() in ['linear', 'lineartransit']:
        print('selecting linear transit model')
        return LinearTransit
    elif model.lower() in ['lld', 'linearlimbdarkening', 'lldtransit']:
        print('selecting Linear Dark-Limbening model')
        return LLDTransit


if __name__ == '__main__':
    N = 100
    time_array = np.linspace(0, 1, N)
    pars = 0.55, 0.1, 0.3, 0.1

    t = LinearTransit(time_array)
    x = np.expand_dims(t.get_flux(time_array=None, transit_pars=pars), 0)
    t.fit(x, sigma=np.random.uniform(0.1, 0.2, N))
    print(np.sqrt(np.diag(t.pcov)))
