import numpy as np

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
