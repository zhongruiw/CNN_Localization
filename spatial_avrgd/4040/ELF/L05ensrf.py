# ----- L05ensrf for elf localization -----
# coding by Zhongrui Wang
# version: 2.6
# update: inde-eva; add rtps inf; git repo; fix bug 481

import sys 
sys.path.append('/home/lllei/AI_localization/L05/git_repo/general')

import numpy as np
from construct_GC_2d import construct_GC_2d
from construct_func_2d import construct_func_2d
from EAKF_wzr import EAKF_wzr
from os.path import dirname, join as pjoin
from scipy.io import loadmat
from step_L04 import step_L04
# import matplotlib.pyplot as plt
# import time


data_dir = '/scratch/lllei/data'
# data_dir = '/Users/ree/Documents/DataAssimilization/hybrid_L05/data'
icsfname = pjoin(data_dir, 'ics_ms3_from_zt1year_sz3001.mat')
obsfname = pjoin(data_dir, 'obs_ms3_err1_240s_6h_25y_sptavd025.mat')
truthfname = pjoin(data_dir, 'zt_25year_ms3_6h.mat')

# get truth
time_steps = 200 * 450  # days(1 day~ 200 time steps)
obs_freq_timestep = 50
obsdens = 4
truthf = loadmat(truthfname)
eva_bg = int(23 * 360 * 200 / obs_freq_timestep)
eva_ed = eva_bg+int(time_steps/obs_freq_timestep+1)
ztruth = truthf['zens_times'][eva_bg:eva_ed, :]

# get initial condition
icsf = loadmat(icsfname)
zics_total = icsf['zics_total1']

# get observation
obsf = loadmat(obsfname)
zobs_total = obsf['zobs_total'][eva_bg:eva_ed, :]

# model parameters
model_size = 960  # N
# observation parameters
obs_density = 4

# regular temporal / sptial obs
model_grids = np.arange(1, model_size + 1)
obs_grids = model_grids[model_grids % obs_density == 0]
nobsgrid = len(obs_grids)

# make forward operator H
Hk = np.mat(np.zeros((nobsgrid, model_size)))
# for iobs in range(0, nobsgrid):
#     x1 = obs_grids[iobs] - 1
#     Hk[iobs, x1] = 1.0

ave_range = 0.25 * model_size
if ave_range % 2 != 0:
    raise ValueError('average range * model_size should be an even number')
else:
    for iobs in range(0, nobsgrid):
        x1 = obs_grids[iobs] - 1
        if x1+int(ave_range/2)+1 > model_size:
            Hk[iobs, x1-int(ave_range/2):model_size] = 1.0 / (ave_range+1)
            Hk[iobs, 0:x1+int(ave_range/2)+1-model_size] = 1.0 / (ave_range+1)
        elif x1-int(ave_range/2) < 0:
            Hk[iobs, 0:x1+int(ave_range/2)+1] = 1.0 / (ave_range+1)
            Hk[iobs, x1-int(ave_range/2):] = 1.0 / (ave_range+1)        
        else:
            Hk[iobs, x1-int(ave_range/2):x1+int(ave_range/2)+1] = 1.0 / (ave_range+1)

zt = ztruth @ Hk.T
print('truth shape:', np.shape(zt))
print('observations error std:', np.std(zt-zobs_total))

# plt.plot(np.arange(0,20), zt[0:20,0])
# plt.plot(np.arange(0,20), zobs_total[0:20,0], '*')


def ensrf(ztruth, zics_total, zobs_total):
    serial_update = 1

    # analysis period
    iobsbeg = 361
    iobsend = int(time_steps/obs_freq_timestep)+1

    # -------------------------------------------------------------------------------
    # eakf parameters
    ensemble_size = 40
    inde_ensemble_size = 40
    ens_mem_beg = 248 # different initials for evaluation
    ens_mem_end = ens_mem_beg + ensemble_size
    inde_ens_mem_beg = 248 # different initials for evaluation
    inde_ens_mem_end = inde_ens_mem_beg + inde_ensemble_size
    
    inflation_value = 1.05
    inflation_value_inde = 1.05
    localize = 1
    localization_value = 600
    localize_inde = 1
    # localization_value_inde = 120
    elff = np.load('/home/lllei/AI_localization/L05/F16/avrgobs/025/obs_dens4/train/elf/iter0_gc/iter0/elf_smsp.npy')
    localize_inde_func = np.concatenate((elff, elff[-2:0:-1]), axis=0)
    # -------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------
    # hybrid parameter
    feedback = 1  # 0 no feedback; 1 feedback
    # -------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------
    # model parameters
    model_size = 960  # N
    forcing = 16.00  # F
    space_time_scale = 10.00  # b
    coupling = 3.00  # c
    smooth_steps = 12  # I
    K = 32
    delta_t = 0.001
    time_step_days = 0
    time_step_seconds = 432
    # time_steps = 200 * 360  # 360 days(1 day~ 200 time steps)
    model_number = 3  # 2 scale
    # -------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------
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
    ave_range = 0.25 * model_size
    if ave_range % 2 != 0:
        raise ValueError('average range * model_size should be an even number')
    else:
        for iobs in range(0, nobsgrid):
            x1 = obs_grids[iobs] - 1
            if x1+int(ave_range/2)+1 > model_size:
                Hk[iobs, x1-int(ave_range/2):model_size] = 1.0 / (ave_range+1)
                Hk[iobs, 0:x1+int(ave_range/2)+1-model_size] = 1.0 / (ave_range+1)
            elif x1-int(ave_range/2) < 0:
                Hk[iobs, 0:x1+int(ave_range/2)+1] = 1.0 / (ave_range+1)
                Hk[iobs, x1-int(ave_range/2):] = 1.0 / (ave_range+1)        
            else:
                Hk[iobs, x1-int(ave_range/2):x1+int(ave_range/2)+1] = 1.0 / (ave_range+1)

    model_times = np.arange(0, time_steps + 1)
    obs_times = model_times[model_times % obs_freq_timestep == 0]
    nobstime = len(obs_times)
    # -------------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------------
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
    # -------------------------------------------------------------------------------
    
    CMat = np.mat(construct_GC_2d(localization_value, model_size, obs_grids))
    CMat_inde = np.mat(construct_func_2d(localize_inde_func, model_size, obs_grids))

    # cmat = np.array(CMat)
    # cmat_inde = np.array(CMat_inde)
    # CMat_state = np.mat(construct_GC_2d(localization_value, model_size, model_grids))

    zens = np.mat(zics_total[ens_mem_beg: ens_mem_end, :])  # ensemble are drawn from ics set
    zens_inde = np.mat(zics_total[inde_ens_mem_beg: inde_ens_mem_end, :])

    prior_spread = np.zeros((model_size, nobstime))
    analy_spread = np.zeros((model_size, nobstime))
    prior_spread_inde = np.zeros((model_size, nobstime))
    analy_spread_inde = np.zeros((model_size, nobstime))

    zeakf_prior = np.zeros((model_size, nobstime))
    # zens_times = np.empty((time_steps+1, model_size))
    zeakf_analy = np.empty((model_size, nobstime))
    zeakf_prior_indeself = np.zeros((model_size, nobstime))
    zeakf_analy_indeself = np.empty((model_size, nobstime))

    # zens_times[0, :] = np.mean(ensemble.z, axis=0)

    for iassim in range(0, nobstime):
        print(iassim)

        # EnKF step
        obsstep = iassim * obs_freq_timestep + 1

        zeakf_prior[:, iassim] = np.mean(zens, axis=0)  # prior ensemble mean
        prior_spread[:, iassim] = np.std(zens, axis=0, ddof=1)
        zeakf_prior_indeself[:, iassim] = np.mean(zens_inde, axis=0)  # prior ensemble mean
        prior_spread_inde[:, iassim] = np.std(zens_inde, axis=0, ddof=1)

        zobs = np.mat(zobs_total[iassim, :])

        # inflation RTPP
        ensmean = np.mean(zens, axis=0)
        ensp = zens - ensmean
        zens = ensmean + ensp * inflation_value

        ensmean_inde = np.mean(zens_inde, axis=0)
        ensp_inde = zens_inde - ensmean_inde
        zens_inde = ensmean_inde + ensp_inde * inflation_value_inde

        # save inflated prior zens
        zens_prior = zens
        zens_inde_prior = zens_inde

        # serial EnSRF update
        zens = EAKF_wzr(1, model_size, ensemble_size, ensemble_size,
                        nobsgrid, zens, zens, Hk, obs_error_var, localize, CMat, zobs)

        zens_inde = EAKF_wzr(1, model_size, inde_ensemble_size, inde_ensemble_size,
                             nobsgrid, zens_inde, zens_inde, Hk, obs_error_var, localize_inde, CMat_inde, zobs)

        # # inflation RTPS
        # std_prior = np.std(zens_prior, axis=0, ddof=1)
        # std_analy = np.std(zens, axis=0, ddof=1)
        # ensmean = np.mean(zens, axis=0)
        # ensp = zens - ensmean
        # zens = ensmean + np.multiply(ensp, (1 + inflation_value*(std_prior-std_analy)/std_analy))
       
        # std_prior = np.std(zens_inde_prior, axis=0, ddof=1)
        # std_analy = np.std(zens_inde, axis=0, ddof=1)
        # ensmean = np.mean(zens_inde, axis=0)
        # ensp = zens_inde - ensmean
        # zens_inde = ensmean + np.multiply(ensp, (1 + inflation_value*(std_prior-std_analy)/std_analy))
       
        # save zens_analy_kg_f
        zens_analy_kg_f = np.mean(zens, axis=0)
        zeakf_analy[:, iassim] = zens_analy_kg_f
        zens_analy_indeself = np.mean(zens_inde, axis=0)
        zeakf_analy_indeself[:, iassim] = zens_analy_indeself

        # save analy_spread
        analy_spread[:, iassim] = np.std(zens, axis=0, ddof=1)
        analy_spread_inde[:, iassim] = np.std(zens_inde, axis=0, ddof=1)

        # check for an explosion
        if False in np.isreal(zens):
            print('Found complex numbers, stopping this run')
            return

        # ensemble model advance, but no advance model from the last obs
        if iassim < nobstime - 1:
            step_beg = obsstep
            step_end = (iassim + 1) * obs_freq_timestep
            # start = time.time()
            for istep in range(step_beg, step_end + 1):
                zens = step_L04(ensemble_size, zens, model_size, ss2, smooth_steps, a, model_number, K4, H, K, K2, sts2, coupling,
                                 space_time_scale, forcing, delta_t)
                zens_inde = step_L04(inde_ensemble_size, zens_inde, model_size, ss2, smooth_steps, a, model_number, K4, H, K, K2, sts2,
                                coupling, space_time_scale, forcing, delta_t)

    zt = ztruth.T
    prior_rmse = np.sqrt(np.mean((zt - zeakf_prior) ** 2, axis=0))
    analy_rmse = np.sqrt(np.mean((zt - zeakf_analy) ** 2, axis=0))
    prior_spread_rmse = np.sqrt(np.mean(prior_spread ** 2, axis=0))
    prior_spread_inde_rmse = np.sqrt(np.mean(prior_spread_inde ** 2, axis=0))
    analy_spread_rmse = np.sqrt(np.mean(analy_spread ** 2, axis=0))
    analy_spread_inde_rmse = np.sqrt(np.mean(analy_spread_inde ** 2, axis=0))

    prior_indeself_rmse = np.sqrt(np.mean((zt - zeakf_prior_indeself) ** 2, axis=0))
    analy_indeself_rmse = np.sqrt(np.mean((zt - zeakf_analy_indeself) ** 2, axis=0))

    prior_err = np.mean(prior_rmse[iobsbeg - 1: iobsend])
    analy_err = np.mean(analy_rmse[iobsbeg - 1: iobsend])
    prior_indeself_err = np.mean(prior_indeself_rmse[iobsbeg - 1: iobsend])
    analy_indeself_err = np.mean(analy_indeself_rmse[iobsbeg - 1: iobsend])

    print('prior error = {0:.6f}'.format(prior_err))
    print('prior elf error = {0:.6f}'.format(prior_indeself_err))
    print('analy gc error = {0:.6f}'.format(analy_err))
    print('analy elf error = {0:.6f}'.format(analy_indeself_err))

    # save
    np.save('prior_rmse_gc240.npy', prior_rmse)
    np.save('prior_rmse_elf.npy', prior_indeself_rmse)
    np.save('analy_rmse_gc240.npy', analy_rmse)
    np.save('analy_rmse_elf.npy', analy_indeself_rmse)
    np.save('prior_spread_rmse.npy', prior_spread_rmse)
    np.save('prior_spread_inde_rmse.npy', prior_spread_inde_rmse)
    np.save('analy_spread_rmse.npy', analy_spread_rmse)
    np.save('analy_spread_inde_rmse.npy', analy_spread_inde_rmse)
    # np.save('zens_times_prior.npy', zens_times)
    # np.save('truth_times.npy', ztruth)
    # np.save('observations.npy', zobs_total)
    np.save('zeakf_prior.npy', zeakf_prior)
    np.save('zeakf_prior_inde.npy', zeakf_prior_indeself)
    np.save('zeakf_analy.npy', zeakf_analy)
    # # np.save('zeakf_analy_adm.npy', zeakf_analy_adm)
    # np.save('zeakf_analy_inde.npy', zeakf_analy_inde)
    np.save('zeakf_analy_indeself.npy', zeakf_analy_indeself)

    return prior_err


err = ensrf(ztruth, zics_total, zobs_total)
