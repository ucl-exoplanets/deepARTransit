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
        p0 = (np.median(time_array),
              0.1,
             (time_array[-1] - time_array[0]) / 2,
             (time_array[-1] - time_array[0]) / 10)
    if bounds is None:
        bounds= ((time_array[0],0.,0.,0.),
                 (time_array[-1],
                  0.5,
                  (time_array[-1] - time_array[0]),
                  (time_array[-1] - time_array[0])/2))

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