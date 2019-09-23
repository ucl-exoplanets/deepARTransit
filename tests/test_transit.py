import numpy as np
from deepartransit.utils.transit import LinearTransit, LLDTransit, QLDTransit

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


pars_LLD = [1.98558305e-01, 1.44291168e-01, 1.00000000e+01, 3.92659972e+00,
       8.02690690e+01, 0.005,1e-10, 4.99910154e-01]

def test_LLDTransit():
    t = LLDTransit(time_array)
    x = np.expand_dims(t.get_flux(time_array=None, transit_pars=pars_LLD), 0)
    t.fit(x, sigma = np.random.uniform(1.01, 1.03, N))
    np.testing.assert_allclose(t.delta, pars_LLD[1]**2, rtol=5e-06)
    #np.testing.assert_allclose(t.transit_pars, pars_LLD)


    assert t.duration < time_array[-1] - time_array[0] and t.duration > time_array[1] - time_array[0]

pars_QLD = [0.64959831, -0.04842008, -0.16233996,  0.07468994, 1.44291168e-01, 1.00000000e+01, 3.92659972e+00,
       8.02690690e+01, 0.005,1e-10, 4.99910154e-01]

def test_QLDTransit():
    t = QLDTransit(time_array)
    x = np.expand_dims(t.get_flux(time_array=None, transit_pars=pars_QLD), 0)
    t.fit(x, sigma = np.random.uniform(1.01, 1.03, N))
    np.testing.assert_allclose(t.delta, pars_QLD[4]**2, rtol=5e-05)
    #np.testing.assert_allclose(t.transit_pars, pars_LLD)
    assert t.duration < time_array[-1] - time_array[0] and t.duration > time_array[1] - time_array[0]