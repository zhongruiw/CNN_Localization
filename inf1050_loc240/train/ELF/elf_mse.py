# ----- calculate elf mse -----
# coding by Zhongrui Wang
# version 2.0
# update : add analysis mse

# import matplotlib.pyplot as plt
import numpy as np
from construct_func_2d import construct_func_2d
from scipy.io import loadmat


model_size = 960
obs_dens = 4
nobs = int(model_size/obs_dens)
loc_value = 240
model_grids = np.arange(1, model_size + 1)
obs_grids = model_grids[model_grids % obs_dens == 0]

# make forward operator H
Hk = np.zeros((nobs, model_size))
for iobs in range(0, nobs):
    x1 = obs_grids[iobs] - 1
    Hk[iobs, x1] = 1.0

elff = np.load('elf_ztest.npy')
localize_inde_func = np.concatenate((elff, elff[-2:0:-1]), axis=0)
CMat_inde = np.transpose(construct_func_2d(localize_inde_func, model_size, obs_grids))

cut_out = 50
kg_f = np.load('/scratch/lllei/inf1050_loc240/200040/kg_f_5y.npy')[cut_out:,:,:]
# kg_t = np.load('/scratch/lllei/inf1050_loc240/200040/kg_t_5y.npy')[cut_out:,:,:]
len_t = kg_f.shape[0]

# se = (kg_f*np.tile(CMat_inde[None,:,:], (len_t,1,1)) - kg_t)**2

# for j in range(0, nobs):
#     se[:, :, j] = np.roll(se[:, :, j], -(obs_dens*j+obs_dens-1), axis=1)

# mse = np.mean(np.mean(se, axis=0),axis=1)

# np.save('mse_elf_ztest.npy', mse)
 

# elff = np.load('elf.npy')
# localize_inde_func = np.concatenate((elff, elff[-2:0:-1]), axis=0)
# CMat_inde = np.transpose(construct_func_2d(localize_inde_func, model_size, obs_grids))

# se = (kg_f*np.tile(CMat_inde[None,:,:], (len_t,1,1)) - kg_t)**2

# for j in range(0, nobs):
#     se[:, :, j] = np.roll(se[:, :, j], -(obs_dens*j+obs_dens-1), axis=1)

# mse = np.mean(np.mean(se, axis=0),axis=1)

# np.save('mse_elf.npy', mse)

## Loss: xt
truthf = loadmat('/scratch/lllei/data/zt_5year_ms3_6h.mat')
ztruth = truthf['zens_times']
xt = ztruth[1:,:]
xf = np.load('/home/lllei/AI_localization/L05/e2000_inf1050_e40_inf1050_loc240/200040/save_interval_6h/zeakf_prior.npy')
xf = xf[:, 1:]
obs = np.load('/home/lllei/AI_localization/L05/e2000_inf1050_e40_inf1050_loc240/200040/save_interval_6h/observations.npy')
obs = obs[1:, :]
incs = obs - (Hk @ xf).T
xf = xf.T

xt = xt[cut_out:,:]
xf = xf[cut_out:,:]
incs = incs[cut_out:,:]

if np.shape(kg_f)[0] != np.shape(xt)[0]:
    raise ValueError('check the dimension consistence')

se = (xf + np.squeeze(kg_f*np.tile(CMat_inde[None,:,:], (len_t,1,1)) @ incs[:,:,None]) - xt)**2
mse = np.squeeze(np.mean(se, axis=0))

print('mse of analysis:', np.mean(mse))
np.save('mse_xt_elf_ztest.npy', mse)