# ------ calculate regression localization ------
# coding by Zhongrui Wang
# version: 1.1
# update: linear model without residual, y=bx

# import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
# from construct_GC_2d import construct_GC_2d


model_size = 960
obs_dens = 4
nobs = int(model_size/obs_dens)
loc_value = 240
model_grids = np.arange(1, model_size + 1)
obs_grids = model_grids[model_grids % obs_dens == 0]
# CMat = construct_GC_2d(loc_value, model_size, obs_grids)

# for j in range(0, nobs):
#     kg_f[:, :, j] = cp.roll(kg_f[:, :, j], -(obs_dens*j+obs_dens-1), axis=1)
#     kg_t[:, :, j] = cp.roll(kg_t[:, :, j], -(obs_dens*j+obs_dens-1), axis=1)

# plt.plot(xx, rolled_kg_f[0,:,4], label='backgroud')
# plt.plot(xx, rolled_kg_t[0,:,4], label='truth')
# plt.legend()
# plt.show()
# plt.savefig('train_history.png')

loc_size = int(model_size/2)
beta = cp.empty(loc_size+1)
# eps = cp.empty(loc_size+1)
# MSE_gc = cp.empty(model_size)
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

    x = kg_f.flatten()  # kg_f[:, i, :].flatten()
    y = kg_t.flatten()  # kg_t[:, i, :].flatten()
    # standardize x,y
    xmean = cp.mean(x)
    ymean = cp.mean(y)

    # revision here
    xp = x
    yp = y

    beta[i] = 1./cp.matmul(xp.T, xp) * cp.matmul(cp.transpose(xp), yp)
    # eps[i] = ymean - beta[i] * xmean

    len_t = cp.shape(kg_f)[0]
    nobsgrid = cp.shape(kg_f)[1]

    # bias = cp.asarray(CMat[0, i]) * x - y
    # MSE_gc[i] = cp.matmul(cp.transpose(bias), bias) / (len_t * nobsgrid)
    # bias = beta[i] * x - y
    bias = beta[i] * x - y
    MSE_reg[i] = cp.matmul(cp.transpose(bias), bias) / (len_t * nobsgrid)

# beta[0] = 1
# x[:,0] = rolled_kg_f[:,0,:].flatten()
# y[:,0] = rolled_kg_t[:,0,:].flatten()

MSE_reg[loc_size+1:] = MSE_reg[loc_size-1:0:-1]

np.save('reg_loc_beta.npy', cp.asnumpy(beta))
# np.save('reg_loc_eps.npy', cp.asnumpy(eps))
# np.save('mse_gc_240.npy', cp.asnumpy(MSE_gc))
np.save('mse_reg.npy', cp.asnumpy(MSE_reg))

# plt.plot(range(model_size), beta, label='regression')
# plt.plot(range(model_size), CMat[0, :], label='GC')
# plt.legend()
# plt.show()
# plt.savefig('loc_func.png')
# 