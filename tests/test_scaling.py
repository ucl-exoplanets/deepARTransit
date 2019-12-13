import numpy as np

from deepartransit.utils.scaling import Scaler, MeanStdScaler, MinMaxScaler


def test_Scalers():
    fixtures = []
    fixtures.append([np.random.uniform(0, 1, size=(100, 6, 5)), None, 0])
    fixtures.append([np.random.uniform(0, 1, size=(100, 1, 5)), None, 0])
    fixtures.append([np.random.uniform(0, 1, size=(7, 100, 5, 5)), None, 1])
    fixtures.append([np.random.uniform(0, 1, size=(100, 6)), None, 0])
    fixtures.append([np.random.uniform(0, 1, size=(100, 6, 5)), range(30), 0])

    fixtures.append([np.ones(shape=(100, 6, 5)), None, 0])

    for plc, train_range, time_axis in fixtures:
        try:
            if train_range is None:
                train_range = range(plc.shape[time_axis])

            scaler = Scaler(train_range, time_axis)
            scaler.fit(plc)

            # MinMaxScaler
            scaler = MinMaxScaler(train_range, time_axis)
            scaler.fit(plc)
            np.testing.assert_allclose(np.zeros(plc.min(time_axis).shape),
                                       scaler.transform(plc).take(train_range, time_axis).min(time_axis),
                                       rtol=1e-5, atol=1e-10)
            np.testing.assert_allclose(np.ones(plc.max(time_axis).shape),
                                       scaler.transform(plc).take(train_range, time_axis).max(time_axis),
                                       rtol=1e-5, atol=1e-10)
            assert scaler.inverse_transform(scaler.transform(plc)).shape == plc.shape
            np.testing.assert_allclose(plc, scaler.inverse_transform(scaler.transform(plc)), rtol=1e-10)

            # MeanStdScaler
            scaler = MeanStdScaler(train_range, time_axis)
            scaler.fit(plc)
            np.testing.assert_allclose(np.zeros(plc.mean(time_axis).shape),
                                       scaler.transform(plc).take(train_range, time_axis).mean(time_axis),
                                       atol=1e-10)
            np.testing.assert_allclose(np.ones(plc.std(time_axis).shape),
                                       scaler.transform(plc).take(train_range, time_axis).std(time_axis), atol=1e-10)
            assert scaler.inverse_transform(scaler.transform(plc)).shape == plc.shape
            np.testing.assert_allclose(plc, scaler.inverse_transform(scaler.transform(plc)), rtol=1e-10)

        except ZeroDivisionError:
            assert scaler.zero_norm
