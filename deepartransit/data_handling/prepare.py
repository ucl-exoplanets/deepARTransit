import numpy as np
import os

if os.getcwd().split('/')[-1] != 'spitzer_light_curves':
    print(os.getcwd().split('/')[-1])
    os.chdir('..')
import utilities as utils
from observation import Observation

def create_dataset(aorkey='22807808', channel=4, radius=2):
    dates, data_array, header = utils.load_data(aorkey, channel)
    obs = Observation(aorkey, channel, header, dates, data_array, radius=radius, planet='none')

    obs.preprocess()
    obs.compute_centroids()

    data = obs.flux
    data = np.expand_dims(data.reshape((data.shape[0], obs.flux.shape[-1] * obs.flux.shape[-2])).T, -1)

    cent_cov = np.concatenate([[obs.x_centroids],
                               [obs.y_centroids]
                               ]).T
    cent_cov = np.expand_dims(cent_cov, 0)
    return data, cent_cov

if __name__ == '__main__':
    aorkey = '22807808'
    data, cent_cov = create_dataset()
    print(data.shape, cent_cov.shape)

    np.save('plc_{}.npy'.format(aorkey), data)
    np.save('cent_{}.npy'.format(aorkey), cent_cov)