import numpy as np



obs_dens = 4
nobs = 240
model_size = 960

kg = np.load('/scratch/lllei/spatially_averaged/200040/avrgd_025/obs_dens4/kg_f_5y.npy')

for j in range(0, nobs):
    kg[:, :, j] = np.roll(kg[:, :, j], -(obs_dens*j+obs_dens-1), axis=1)

for i in range(0, model_size):
    stri = 'kg_f_5y_{:d}.npy'
    np.save(stri.format(i), kg[:, i, :])

kg = np.load('/scratch/lllei/spatially_averaged/200040/avrgd_025/obs_dens4/kg_t_5y.npy') 

for j in range(0, nobs):
    kg[:, :, j] = np.roll(kg[:, :, j], -(obs_dens*j+obs_dens-1), axis=1)

for i in range(0, model_size):
    stri = 'kg_t_5y_{:d}.npy'
    np.save(stri.format(i), kg[:, i, :])