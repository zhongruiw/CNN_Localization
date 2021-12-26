# ------ calculate regression localization 2d -------
# coding by Zhongrui Wang
# version : 2.0
# update: git repo; without residual

# import matplotlib.pyplot as plt
import numpy as np
# import cupy as cp


model_size = 960
obs_dens = 4
nobs = int(model_size/obs_dens)
model_grids = np.arange(1, model_size + 1)
obs_grids = model_grids[model_grids % obs_dens == 0]
loc_size = nobs

beta = np.empty((model_size,loc_size))
eps = np.empty((model_size,loc_size))
MSE_reg = np.empty((model_size,loc_size))

kg_f = np.load('/scratch/lllei/inf1050_loc240/200040/kg_f_5y.npy')
kg_t = np.load('/scratch/lllei/inf1050_loc240/200040/kg_t_5y.npy')

for i in range(0, model_size):
    for j in range(0, loc_size):
        x = kg_f[:,i,j]
        y = kg_t[:,i,j]
        # standardize x,y
        xp = x
        yp = y

        beta[i,j] = 1./np.matmul(xp.T, xp) * np.matmul(np.transpose(xp), yp)

        len_t = np.shape(kg_f)[0]

        bias = beta[i,j] * x - y
        MSE_reg[i,j] = np.matmul(np.transpose(bias), bias) / len_t

np.save('reg_2d_beta.npy', beta)
np.save('mse_reg_2d.npy', MSE_reg)
