import numpy as np


kg = np.load('/home/lllei/AI_localization/L05/1y/server/e2000_inf1050_e40_inf1050_loc240/200040/save_interval_6h/kg_f_rolled.npy')  # np.load('/Volumes/TOSHIBA EXT/backup/kg_f_1y.npy')

obs_dens = 4
nobs = 240
model_size = 960
times = 7200

for i in range(0, 10):
    stri = 'kg_f_rolled_{:d}.npy'
    np.save(stri.format(i), kg[(i*720):(i*720+720), :, :])

kg = np.load('/home/lllei/AI_localization/L05/1y/server/e2000_inf1050_e40_inf1050_loc240/200040/save_interval_6h/kg_t_rolled.npy')  # np.load('/Volumes/TOSHIBA EXT/backup/kg_f_1y.npy')

for i in range(0, 10):
    stri = 'kg_t_rolled_{:d}.npy'
    np.save(stri.format(i), kg[(i*720):(i*720+720), :, :])