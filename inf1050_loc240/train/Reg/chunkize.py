import numpy as np



obs_dens = 4
nobs = 240
model_size = 960

kg = np.load('/scratch/lllei/inf1050_loc240/200040/kg_f_rolled.npy')

for i in range(0, model_size):
    stri = 'kg_f_5y_{:d}.npy'
    np.save(stri.format(i), kg[:, i, :])

kg = np.load('/scratch/lllei/inf1050_loc240/200040/kg_t_rolled.npy') 

for i in range(0, model_size):
    stri = 'kg_t_5y_{:d}.npy'
    np.save(stri.format(i), kg[:, i, :])