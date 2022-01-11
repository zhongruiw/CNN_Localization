# ----- calculate localization & mse -----
# coding by Zhongrui Wang
# version 1.2
# update: add srcnn_dnn_xt

import numpy as np
from os.path import dirname, join as pjoin

# optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import InputSpec 
from tensorflow.python.keras.utils import conv_utils
from give_a_model import srcnn
# from construct_func_2d import construct_func_2d


# ------------------ experiment settings ------------------
model_size = 960
obs_dens = 4
nobs = int(model_size / obs_dens)
model_grids = np.arange(1, model_size + 1)
obs_grids = model_grids[model_grids % obs_dens == 0]

# ---------------------- load data ------------------------
datadir = '/scratch/lllei/inf1050_loc240/200040/'
stri = pjoin(datadir, 'kg_f_5y.npy')
kg_f = np.load(stri)
stri = pjoin(datadir, 'kg_t_5y.npy')
kg_t = np.load(stri)

# cut off 
cut_off = 41
kg_f = kg_f[cut_off:,:,:]
kg_t = kg_t[cut_off:,:,:]

# ------------------ load model (optional)-----------------
model = srcnn(batch_size=1)
model.load_weights('./my_weights/srcnn')

# intermediate layer output (optional)
layer_name = 'loc_f'
interm_layer_model = keras.Model(inputs=model.input,
                                       outputs=model.get_layer(layer_name).output)

# roll kg (optional)
# for j in range(0, nobs):
#     kg_f[:, :, j] = np.roll(kg_f[:, :, j], -(obs_dens*j+obs_dens-1), axis=1)
#     kg_t[:, :, j] = np.roll(kg_t[:, :, j], -(obs_dens*j+obs_dens-1), axis=1)

# chunkize kg (optional)
# for i in range(0, model_size):
#     stri = 'kg_f_5y_{:d}.npy'
#     np.save(stri.format(i), kg_f[:, i, :])
#     stri = 'kg_t_5y_{:d}.npy'
#     np.save(stri.format(i), kg_t[:, i, :])

# ---------------- choose localization schemes -------------
# 1D localization function
## with residual
# beta = np.load('/home/lllei/AI_localization/L05/e2000_inf1050_e40_inf1050_loc240/train/ai/CNN/SRCNN_DNN/distance_only/debug_481/loc_srcnn_dnn.npy')
# beta = np.concatenate((beta, beta[-2:0:-1]), axis=0)
# CMat = construct_func_2d(beta, model_size, obs_grids)
# eps = np.load('/home/lllei/AI_localization/L05/e2000_inf1050_e40_inf1050_loc240/train/reg/debug_481/reg_loc_eps.npy')
# eps = np.concatenate((eps, eps[-2:0:-1]), axis=0)
# CMat_eps = construct_func_2d(eps, model_size, obs_grids)

# ntime = kg_f.shape[0]
# kg_pred = kg_f * np.tile(CMat[None,:,:], (ntime,1,1)) + np.tile(CMat_eps[None,:,:], (ntime,1,1))

## without residual 
# beta = np.load('/home/lllei/AI_localization/L05/e2000_inf1050_e40_inf1050_loc240/train/ai/CNN/SRCNN_DNN/distance_only/debug_481/loc_srcnn_dnn.npy')
# beta = np.concatenate((beta, beta[-2:0:-1]), axis=0)
# CMat = construct_func_2d(beta, model_size, obs_grids)

# ntime = kg_f.shape[0]
# kg_pred = kg_f * np.tile(CMat[None,:,:], (ntime,1,1))

# srcnn
# kg_pred = np.squeeze(model.predict_on_batch(np.reshape(kg_f,(-1,model_size,nobs,1))))
# kg_pred = np.zeros(kg_f.shape)
# for i in range(10):
#     kg_pred[715*i:715*(i+1),:,:] = np.squeeze(model.predict(np.reshape(kg_f[715*i:715*(i+1),:,:],(-1,model_size,nobs,1)), batch_size=8))

# srcnn_dnn 
loc_size = 481
# loc_ai = np.zeros((kg_f.shape[0],loc_size))
loc_ai = np.squeeze(interm_layer_model.predict(np.reshape(kg_f, (-1,model_size,nobs,1)), batch_size=8))

# -------------------- calculate mse --------------------
# localized kgain mse
# se = np.square(kg_pred - kg_t)

# for j in range(0, nobs):
#     se[:, :, j] = np.roll(se[:, :, j], -(obs_dens*j+obs_dens-1), axis=1)

# mse = np.mean(np.mean(se,axis=0),axis=1)
# np.save('mse_srcnn_dnn_xt.npy', mse)

# ----------------- calculte localization -----------------
# srcnn-static localization
# loc = np.mean(kg_pred / kg_f, axis=0)
# np.save('loc_srcnn.npy', loc)

# srcnn_dnn
loc_ai_mean = np.mean(loc_ai, axis=0)
np.save('loc_srcnn_dnn_xt.npy', loc_ai_mean)
