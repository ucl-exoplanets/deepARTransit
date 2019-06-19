import numpy as np
import scipy.optimize as opt

#TODO: class transit linear
#TODO: tensorflow implementation

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
def fit_transit_linear(data, time_array=None, repeat=1, length_pred=None):
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
    popt, pcov = opt.curve_fit(transit_linear,
                              np.expand_dims(time_array, 0).repeat(repeat, 0).flatten(),
                              data.flatten(),
                              p0=(0.5, 0.1, 0.5, 0.05),
                              bounds= ((0.,0.,0.,0.), (1.,0.5, 1., 0.5)),
                              maxfev=100000)
    return popt