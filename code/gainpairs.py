# ----- generate noisy-true gain pairs -----
# version: 1.0

import numpy as np
from construct_GC_2d import construct_GC_2d
from EnKF import EnKF
from os.path import dirname, join as pjoin
from scipy.io import loadmat
from step_L04 import step_L04


np.random.seed(2022)

# load data
data_dir = '../data/assimilation'
icsfname = pjoin(data_dir, 'ics_ms3_from_zt1year_sz3001.mat')
obsfname = pjoin(data_dir, 'obs_ms3_err1_240s_6h_25y.mat')
truthfname = pjoin(data_dir, 'zt_25year_ms3_6h.mat')

# get truth
time_steps = 200 * 360 * 5  # 360 days(1 day~ 200 time steps)
obs_freq_timestep = 50
obsdens = 4
truthf = loadmat(truthfname)
ztruth = truthf['zens_times']

# get initial condition
icsf = loadmat(icsfname)
zics_total = icsf['zics_total1']

# get observation
obsf = loadmat(obsfname)
zobs_total = obsf['zobs_total'][0:int(time_steps/obs_freq_timestep+1), :]

# analysis period
iobsbeg = 41
iobsend = int(time_steps/obs_freq_timestep)+1

# hybrid parameter
feedback = 1  # 0 no feedback; 1 feedback

# model parameters
model_size = 960  # N
forcing = 15.00  # F
space_time_scale = 10.00  # b
coupling = 3.00  # c
smooth_steps = 12  # I
K = 32
delta_t = 0.001
time_step_days = 0
time_step_seconds = 432
model_number = 3  # 2 scale

# observation parameters
obs_density = 4
obs_error_var = 1.0

# regular temporal / sptial obs
model_grids = np.arange(1, model_size + 1)
obs_grids = model_grids[model_grids % obs_density == 0]
nobsgrid = len(obs_grids)

R = np.mat(obs_error_var * np.eye(nobsgrid, nobsgrid))

# make forward operator H
Hk = np.mat(np.zeros((nobsgrid, model_size)))
for iobs in range(0, nobsgrid):
    x1 = obs_grids[iobs] - 1
    Hk[iobs, x1] = 1.0

model_times = np.arange(0, time_steps + 1)
obs_times = model_times[model_times % obs_freq_timestep == 0]
nobstime = len(obs_times)

# for speed
H = int(K / 2)
K2 = 2 * K
K4 = 4 * K
ss2 = 2 * smooth_steps
sts2 = space_time_scale ** 2

# smoothing filter
alpha = (3.0 * (smooth_steps ** 2) + 3.0) / (2.0 * (smooth_steps ** 3) + 4.0 * smooth_steps)
beta = (2.0 * (smooth_steps ** 2) + 1.0) / (1.0 * (smooth_steps ** 4) + 2.0 * (smooth_steps ** 2))
ri = - smooth_steps - 1.0
j = 0
a = np.zeros(2*smooth_steps+1)
for i in range(-smooth_steps, smooth_steps+1):
    ri = ri + 1.0
    a[j] = alpha - beta * abs(ri)
    j = j + 1
a[0] = a[0] / 2.00
a[2 * smooth_steps] = a[2 * smooth_steps] / 2.00

# enkf parameters
ensemble_size = 40
ensemble_size_2000 = 2000
ens_mem_beg = 1
ens_mem_end = ens_mem_beg + ensemble_size
ens_mem_beg_2000 = 1
ens_mem_end_2000 = ens_mem_beg_2000 + ensemble_size_2000

inflation_value = 1.01
inflation_value_2000 = 1.01 
localize = 1
localization_value = 240 # localization length scale for 40 ensembles
localize_2000 = 0 # 2000 ensembles without localization

CMat = np.mat(construct_GC_2d(localization_value, model_size, obs_grids))

zens = np.mat(zics_total[ens_mem_beg: ens_mem_end, :])  # ensemble are drawn from ics set
zens_2000 = np.mat(zics_total[ens_mem_beg_2000: ens_mem_end_2000, :])

# save settings
save_interval = obs_freq_timestep
save_num = int(time_steps / save_interval)
kg_f = np.zeros((save_num, model_size, nobsgrid))
kg_t = np.zeros((save_num, model_size, nobsgrid))
isave = 0

zeakf_prior = np.zeros((model_size, nobstime))
zeakf_prior_2000 = np.zeros((model_size, nobstime))
zens_times = np.empty((time_steps+1, model_size))
zeakf_analy = np.empty((model_size, nobstime))
zeakf_analy_2000 = np.empty((model_size, nobstime))

for iassim in range(0, nobstime):
    print(iassim)
    # EnKF step
    obsstep = iassim * obs_freq_timestep + 1

    zeakf_prior[:, iassim] = np.mean(zens, axis=0)  # prior ensemble mean
    zeakf_prior_2000[:, iassim] = np.mean(zens_2000, axis=0)  # prior ensemble mean

    # perturn observations
    zobs = np.mat(zobs_total[iassim, :])
    obs_p = np.random.normal(0., np.sqrt(obs_error_var), (ensemble_size,nobsgrid))
    obs = zobs + obs_p
    
    # inflation
    ensmean = np.mean(zens, axis=0)
    ensp = zens - ensmean
    zens = ensmean + ensp * inflation_value
 
    ensmean_2000 = np.mean(zens_2000, axis=0)
    ensp_2000 = zens_2000 - ensmean_2000
    zens_2000 = ensmean_2000 + ensp_2000 * inflation_value_2000

    # serial EnKF update
    zens = EnKF(model_size, ensemble_size,
                    nobsgrid, zens, Hk, obs_error_var, localize, CMat, obs)
    zens_2000 = EnKF(model_size, ensemble_size_2000,
                    nobsgrid, zens_2000, Hk, obs_error_var, localize_2000, CMat, obs)

    # save
    zens_analy = np.mean(zens, axis=0)
    zeakf_analy[:, iassim] = zens_analy
    zens_analy_2000 = np.mean(zens_2000, axis=0)
    zeakf_analy_2000[:, iassim] = zens_analy_2000

    # check for an explosion
    if False in np.isreal(zens):
        print('Found complex numbers, stopping this run')
        return

    # ensemble model advance, but no advance model from the last obs
    if iassim < nobstime - 1:
        step_beg = obsstep
        step_end = (iassim + 1) * obs_freq_timestep
        for istep in range(step_beg, step_end + 1):
            zens = step_L04(ensemble_size, zens, model_size, ss2, smooth_steps, a, model_number, K4, H, K, K2, sts2, coupling,
                             space_time_scale, forcing, delta_t)
            zens_2000 = step_L04(ensemble_size_2000, zens_2000, model_size, ss2, smooth_steps, a, model_number, K4, H, K, K2, sts2,
                    coupling, space_time_scale, forcing, delta_t)

            if istep % save_interval == 0:
                zens_tmp = zens
                zens_2000_tmp = zens_2000
                
                # inflation
                ensmean = np.mean(zens_tmp, axis=0)
                ensp = zens_tmp - ensmean
                zens_tmp = ensmean + ensp * inflation_value

                ensmean_2000 = np.mean(zens_2000_tmp, axis=0)
                ensp_2000 = zens_2000_tmp - ensmean_2000
                zens_2000_tmp = ensmean_2000 + ensp_2000 * inflation_value_2000

                rn = 1.0 / (ensemble_size - 1)
                Xprime = zens_tmp - np.mean(zens_tmp, axis=0)
                HXens = (Hk * zens_tmp.T).T
                HXprime = HXens - np.mean(HXens, axis=0)
                PbHt = (Xprime.T * HXprime) * rn
                HPbHt = (HXprime.T * HXprime) * rn
                kg_f[isave, :, :] = PbHt * (HPbHt + R).I

                zens_full = zens_2000_tmp
                rn = 1.0 / (ensemble_size_2000 - 1)
                zens_full_mean = np.mean(zens_full, axis=0)
                Xprime_full = zens_full - zens_full_mean
                HXens_full = (Hk * zens_full.T).T
                HXprime_full = HXens_full - (Hk * zens_full_mean.T).T
                PbHt_full = (Xprime_full.T * HXprime_full) * rn
                HPbHt_full = (HXprime_full.T * HXprime_full) * rn
                kg_t[isave, :, :] = PbHt_full * (HPbHt_full + R).I

                isave = isave + 1
zt = ztruth.T

prior_rmse = np.sqrt(np.mean((zt - zeakf_prior) ** 2, axis=0))
analy_rmse = np.sqrt(np.mean((zt - zeakf_analy) ** 2, axis=0))
prior_err = np.mean(prior_rmse[iobsbeg - 1: iobsend])
analy_err = np.mean(analy_rmse[iobsbeg - 1: iobsend])

prior_2000_rmse = np.sqrt(np.mean((zt - zeakf_prior_2000) ** 2, axis=0))
analy_2000_rmse = np.sqrt(np.mean((zt - zeakf_analy_2000) ** 2, axis=0))
prior_2000_err = np.mean(prior_2000_rmse[iobsbeg - 1: iobsend])
analy_2000_err = np.mean(analy_2000_rmse[iobsbeg - 1: iobsend])

print('prior error ensemble size 40= {0:.6f}'.format(prior_err))
print('prior error ensemble size 2000 = {0:.6f}'.format(prior_2000_err))
print('analy error ensemble size 40 = {0:.6f}'.format(analy_err))
print('analy error ensemble size 2000 = {0:.6f}'.format(analy_2000_err))

# save
np.save('kg_f_5y.npy', kg_f)
np.save('kg_t_5y.npy', kg_t)
