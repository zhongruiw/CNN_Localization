import numpy as np


kg = np.load('kg_f_5y.npy')

obs_dens = 4
nobs = 240
model_size = 960

for j in range(0, nobs):
    kg[:, :, j] = np.roll(kg[:, :, j], -(obs_dens*j+obs_dens-1), axis=1)

np.save('kg_f_rolled.npy', kg)
