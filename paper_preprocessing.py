import numpy as np
import sys
import os
print(os.getcwd())
from utils.transit import LinearTransit
sys.path.append('../spitzerLC/')
from spitzerlc import data_handling
from spitzerlc.observation import Observation
from main import create_dataset


# Transits aorkeys
aorkeys = ['22807296', '22807552', '22807808', '24537856', '27603712', '27773440']



plc_array, rlc_array, cent_array = create_dataset(aorkeys, 4, radius=2,
                                                  data_dir = '../spitzerLC/spitzerlc/data/agol_hd189733b',
                                                  background_removal=False)



len_seq = 690
transit = LinearTransit(np.arange(len_seq), [250.3, 0.015, 60, 10])
save_dir = 'deepartransit/data/agol_189733b_r2/'


# for i in range(len(aorkeys)):
#     print(aorkeys[i])
#     name_rlc = 'rlc_' + aorkeys[i]
#     name_cent = 'cent_' + aorkeys[i]
#
#     rlc = np.expand_dims(rlc_array[i], 0)[:,:len_seq]
#     cent = np.expand_dims(cent_array[i, :len_seq], 0)
#     np.save(save_dir + name_rlc, rlc)
#     np.save(save_dir + name_cent, cent)



rlc_array = rlc_array[:, :len_seq]
cent_array = cent_array[:, :len_seq]

np.save(save_dir + 'rlc_all', rlc_array)
np.save(save_dir + 'cent_all', cent_array)

# name_rlc = 'rlc_artif_all'
# rlc_array *= np.expand_dims(np.expand_dims(transit.flux, 0), -1)
# np.save(save_dir + name_rlc, rlc_array)
#
# for i in range(len(aorkeys)):
#     print(aorkeys[i])
#     rlc = np.expand_dims(rlc_array[i], 0)
#
#     name_rlc = 'rlc_artif_' + aorkeys[i]
#     np.save(save_dir + name_rlc, rlc)