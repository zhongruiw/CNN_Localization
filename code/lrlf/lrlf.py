# ------ calculate regression localization ------
# version: 1.1
# update: linear model without residual, y=bx

import numpy as np
import cupy as cp

model_size = 960
obs_dens = 4
nobs = int(model_size/obs_dens)
loc_value = 240
model_grids = np.arange(1, model_size + 1)
obs_grids = model_grids[model_grids % obs_dens == 0]

loc_size = int(model_size/2)
beta = cp.empty(loc_size+1)
MSE_reg = cp.empty(model_size)

for i in range(0, loc_size+1):
    strf = 'kg_f_5y_{:d}.npy'
    strt = 'kg_t_5y_{:d}.npy'
    if i == 0 or i == loc_size:
        kg_f = cp.asarray(np.load(strf.format(i))[41:, :])
        kg_t = cp.asarray(np.load(strt.format(i))[41:, :])
    else:
        kg_f1 = np.load(strf.format(i))[41:,:]
        kg_f2 = np.load(strf.format(model_size-i))[41:,:]
        kg_f = cp.asarray(np.concatenate((kg_f1, kg_f2), axis=0))
        kg_t1 = np.load(strt.format(i))[41:,:]
        kg_t2 = np.load(strt.format(model_size-i))[41:,:]
        kg_t = cp.asarray(np.concatenate((kg_t1, kg_t2), axis=0))

    x = kg_f.flatten() 
    y = kg_t.flatten()

    xp = x
    yp = y
    beta[i] = 1./cp.matmul(xp.T, xp) * cp.matmul(cp.transpose(xp), yp)

    len_t = cp.shape(kg_f)[0]
    nobsgrid = cp.shape(kg_f)[1]

np.save('reg_loc_beta.npy', cp.asnumpy(beta))
