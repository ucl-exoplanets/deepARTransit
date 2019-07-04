import numpy as np
import scipy.optimize as opt

#TODO: class transit linear
#TODO: tensorflow implementation



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

    def fit(self, data, p0=None, bounds=None, range_fit=None, time_axis = 1, replace_pars=True):
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
                                   p0=self.p0,
                                   bounds=self.bounds,
                                   maxfev=100000)
        self.popt = popt
        self.pcov = pcov
        if replace_pars:
            self.transit_pars = popt
        return 0

    flux = property(get_flux)

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
                  duration/3))
        else:
            self.bounds = bounds

    @staticmethod
    def _compute_flux(time_array, t_c, delta, T, tau):
        x1 = time_array - (t_c - T / 2 - tau / 2)
        x2 = time_array - (t_c - T / 2 + tau / 2)
        x3 = time_array - (t_c + T / 2 - tau / 2)
        x4 = time_array - (t_c + T / 2 + tau / 2)
        return 1. - delta / tau * (np.maximum(x1, 0.) - np.maximum(x2, 0.) - np.maximum(x3, 0.) + np.maximum(x4, 0.))




def transit_linear(time_array, t_c, delta, T, tau):
    """
    time_array: numpy array of times
    t_c: time of mid_transit
    delta: transit (max) depth
    T: transit duration between mid-ingress and mid-egress
    tau: ingress or egress duration
    """
    x1 = time_array - (t_c - T/2 - tau/2)
    x2 = time_array - (t_c - T/2 + tau/2)
    x3 = time_array - (t_c + T/2 - tau/2)
    x4 = time_array - (t_c + T/2 + tau/2)
    return 1. - delta/tau * (np.maximum(x1, 0.) - np.maximum(x2, 0.) - np.maximum(x3, 0.) + np.maximum(x4, 0.))

# TODO: test (especially for shapes)
def fit_transit_linear(data, time_array=None, repeat=1, length_pred=None, p0=None, bounds=None):
    """
    :param data:
    :param time_array:
    :param repeat: several LC to fit simulatenously to the same params
    :param length_pred:
    :return: (t_c, delta, T, tau)
    """
    if length_pred is None:
        length_pred = data.shape[1]
    if time_array is None:
        time_array = np.linspace(0,1,length_pred)
    if p0 is None:
        p0 = (np.median(time_array), 0.01, 0, 0)
    if bounds is None:
        bounds= ((time_array[0],0.,0.,0.),
                 (time_array[-1],
                  0.5,
                  (time_array[-1] - time_array[0]),
                  (time_array[-1] - time_array[0])/3))

    popt, pcov = opt.curve_fit(transit_linear,
                              np.expand_dims(time_array, 0).repeat(repeat, 0).flatten(),
                              data.flatten(),
                              p0=p0,
                              bounds= bounds,
                              maxfev=100000)
    return popt

if __name__ == '__main__':
    N = 100
    time_array = np.linspace(0, 1, N)
    pars = 0.55, 0.1, 0.6, 0.1

    x = transit_linear(time_array, *pars)
    x = np.expand_dims(x, 0)
    pred = fit_transit_linear(x, time_array)
    import matplotlib.pylab as plt
    print(pred)
    plt.plot(pred)
    print(x.shape)
    plt.plot(x, color='red')
    plt.show()