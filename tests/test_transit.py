import numpy as np
from utils.transit import transit_linear, fit_transit_linear

N = 100
time_array = np.linspace(0, 1, N)
pars = 0.55, 0.1, 0.6, 0.1


def test_transit_linear():
    x = transit_linear(time_array, *pars)
    x = np.expand_dims(x, 0)
    pred = fit_transit_linear(x, time_array)
    np.testing.assert_allclose(pred, pars)
