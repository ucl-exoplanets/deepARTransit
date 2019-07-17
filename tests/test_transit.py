import numpy as np
from utils.transit import LinearTransit

N = 100
time_array = np.linspace(0, 1, N)
pars = 0.55, 0.1, 0.3, 0.1


def test_LinearTransit_1D_data():
    t = LinearTransit(time_array, pars)
    x = t.flux
    t.fit(x, time_axis=0)
    np.testing.assert_allclose(t.transit_pars, pars)
    assert np.any(np.not_equal(t.transit_pars,pars))

def test_LinearTransit_2D_data():
    t = LinearTransit(time_array, pars)
    x = np.expand_dims(t.flux, 0)
    t.fit(x)
    print(t.transit_pars)
    np.testing.assert_allclose(t.transit_pars, pars)

def test_LinearTransit_range():
    t = LinearTransit(time_array)
    x = np.expand_dims(t.get_flux(time_array=None, transit_pars=pars), 0)
    t.fit(x, range_fit=range(10, N-10))
    print(t.transit_pars)
    np.testing.assert_allclose(t.transit_pars, pars)

def test_LinerTransit_duration():
    t = LinearTransit(time_array, pars)
    d = t.duration
    np.testing.assert_allclose(d, pars[-2] + 2 * pars[-1])

def test_sigma():
    t = LinearTransit(time_array)
    x = np.expand_dims(t.get_flux(time_array=None, transit_pars=pars), 0)
    t.fit(x, sigma = np.random.uniform(1.01, 1.03, N))
    np.testing.assert_allclose(t.transit_pars, pars)
    print(t.pcov, np.diag(t.pcov))
