# ----- elf localization funcation -----
# coding by Zhongrui Wang
# version: 2.5
# update: multiple inf,loc;git repo; allow deflation; fix bug, distance 481,not 480

import sys 
sys.path.append('/home/lllei/AI_localization/L05/git_repo/general')

from numba import jit
import numpy as np
from construct_GC_2d import construct_GC_2d
from EAKF_wzr import EAKF_wzr
from os.path import dirname, join as pjoin
from scipy.io import loadmat
from step_L04 import step_L04

# from time import time


data_dir = '/scratch/lllei/data'
# data_dir = '/Users/ree/Documents/DataAssimilization/AI_localization/L05/data'
icsfname = pjoin(data_dir, 'ics_ms3_from_zt1year_sz3001.mat')
obsfname = pjoin(data_dir, 'obs_ms3_err1_240s_6h_5y.mat')
# truobsfname = pjoin(data_dir, 'truth_ms3_240s_6h_5y.mat')
truthfname = pjoin(data_dir, 'zt_5year_ms3_6h.mat')

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
zobs_total = obsf['zobs_total'][0:int(time_steps / obs_freq_timestep + 1), :]  # 360 days

# model parameters
model_size = 960  # N
# observation parameters
obs_density = 4
obs_error_var = 1.0

# regular temporal / sptial obs
model_grids = np.arange(1, model_size + 1)
obs_grids = model_grids[model_grids % obs_density == 0]
nobsgrid = len(obs_grids)

# make forward operator H
Hk = np.mat(np.zeros((nobsgrid, model_size)))
for iobs in range(0, nobsgrid):
    x1 = obs_grids[iobs] - 1
    Hk[iobs, x1] = 1.0

zt = ztruth @ Hk.T
print('truth shape:', np.shape(zt))
print('observations error std:', np.std(zt - zobs_total))


# print('true observations error std:', np.std(ztobs-zt))

def ensrf(ztruth, zics_total, zobs_total):
    serial_update = 1

    # analysis period
    iobsbeg = 41
    iobsend = int(time_steps / obs_freq_timestep) + 1

    # -------------------------------------------------------------------------------
    # eakf parameters
    ensemble_size = 40
    ens_mem_beg = 1
    ens_mem_end = ens_mem_beg + ensemble_size

    inflation_value = 0.4
    localize = 1
    localization_value = 240
    # -------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------
    # hybrid parameter
    feedback = 1  # 0 no feedback; 1 feedback
    # -------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------
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
    for iobs in range(0, nobsgrid):
        x1 = obs_grids[iobs] - 1
        Hk[iobs, x1] = 1.0

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
    a = np.zeros(2 * smooth_steps + 1)
    for i in range(-smooth_steps, smooth_steps + 1):
        ri = ri + 1.0
        a[j] = alpha - beta * abs(ri)
        j = j + 1

    a[0] = a[0] / 2.00
    a[2 * smooth_steps] = a[2 * smooth_steps] / 2.00
    # -------------------------------------------------------------------------------

    CMat = np.mat(construct_GC_2d(localization_value, model_size, obs_grids))
    # CMat_state = np.mat(construct_GC_2d(localization_value, model_size, model_grids))

    zens = np.mat(zics_total[ens_mem_beg: ens_mem_end, :])  # ensemble are drawn from ics set

    # analy_err_kg_f = np.zeros(nobstime)
    # analy_err_kg_t = np.zeros(nobstime)
    # analy_err_gc = np.zeros(nobstime)
    prior_spread = np.zeros((model_size, nobstime))
    analy_spread = np.zeros((model_size, nobstime))

    zeakf_prior = np.zeros((model_size, nobstime))
    zens_times = np.empty((time_steps + 1, model_size))
    zeakf_analy = np.empty((model_size, nobstime))

    # zens_times[0, :] = np.mean(ensemble.z, axis=0)
    nume = np.zeros((int(model_size / 2)+1))
    deno = np.zeros((int(model_size / 2)+1))
    elf = np.zeros((int(model_size / 2)+1))

    ibegin = 100  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    inc_p = np.empty((0, int(model_size / 2)+1))
    inc_t = np.empty((0, int(model_size / 2)+1))

    for iassim in range(0, nobstime):
        print(iassim)

        # EnKF step
        obsstep = iassim * obs_freq_timestep + 1

        zeakf_prior[:, iassim] = np.mean(zens, axis=0)  # prior ensemble mean
        prior_spread[:, iassim] = np.std(zens, axis=0, ddof=1)

        zobs = np.mat(zobs_total[iassim, :])

        # # inflation RTPP
        # ensmean = np.mean(zens, axis=0)
        # ensp = zens - ensmean
        # zens = ensmean + ensp * inflation_value

        # save inflated prior zens
        zens_prior = zens

        if iassim > ibegin:
            rn = 1.0 / (ensemble_size - 1)
            Xprime = zens - np.mean(zens, axis=0)

            Pb = (Xprime.T * Xprime) * rn

            # calculate increments
            zens_prior_mean = np.mean(zens, axis=0)
            # HXmean = np.array(Hk * zens_prior_mean.T)

            zt = np.array(ztruth[iassim, :])
            zp = np.squeeze(np.array(zens_prior_mean))
            nume, deno, iinc_t, iinc_p = addelf(nume, deno, zt, zp,
                                                model_size, nobsgrid, obs_density, obs_error_var, Pb, Hk)
            inc_t = np.append(inc_t, iinc_t, axis=0)
            inc_p = np.append(inc_p, iinc_p, axis=0)

        # serial EnSRF update
        zens = EAKF_wzr(1, model_size, ensemble_size, ensemble_size,
                        nobsgrid, zens, zens, Hk, obs_error_var, localize, CMat, zobs)

        # inflation RTPS
        std_prior = np.std(zens_prior, axis=0, ddof=1)
        std_analy = np.std(zens, axis=0, ddof=1)
        ensmean = np.mean(zens, axis=0)
        ensp = zens - ensmean
        zens = ensmean + np.multiply(ensp, (1 + inflation_value*(std_prior-std_analy)/std_analy))
       
        # save zens_analy_kg_f
        zens_analy_kg_f = np.mean(zens, axis=0)
        zeakf_analy[:, iassim] = zens_analy_kg_f

        # save analy_spread
        analy_spread[:, iassim] = np.std(zens, axis=0, ddof=1)

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
                zens = step_L04(ensemble_size, zens, model_size, ss2, smooth_steps, a, model_number, K4, H, K, K2, sts2,
                                coupling,
                                space_time_scale, forcing, delta_t)

    # zt = ztruth[0: time_steps + 1: obs_freq_timestep, :].T
    zt = ztruth.T

    elf = nume / deno

    prior_rmse = np.sqrt(np.mean((zt - zeakf_prior) ** 2, axis=0))
    analy_rmse = np.sqrt(np.mean((zt - zeakf_analy) ** 2, axis=0))
    prior_spread_rmse = np.sqrt(np.mean(prior_spread ** 2, axis=0))
    analy_spread_rmse = np.sqrt(np.mean(analy_spread ** 2, axis=0))

    prior_err = np.mean(prior_rmse[iobsbeg - 1: iobsend])
    analy_err = np.mean(analy_rmse[iobsbeg - 1: iobsend])

    print('prior error = {0:.6f}'.format(prior_err))
    print('analysis error = {0:.6f}'.format(analy_err))

    # save
    np.save('elf.npy', elf)
    np.save('/scratch/lllei/inf1050_loc240/train/elf/inf1000/inc_t.npy', inc_t)
    np.save('/scratch/lllei/inf1050_loc240/train/elf/inf1000/inc_p.npy', inc_p)
    np.save('prior_rmse.npy', prior_rmse)
    np.save('analy_rmse.npy', analy_rmse)
    np.save('prior_spread_rmse.npy', prior_spread_rmse)
    np.save('analy_spread_rmse.npy', analy_spread_rmse)

    return prior_err


@jit(nopython=True)
def covind(d, p, obs_density, model_size):
    if obs_density * p + obs_density - 1 + d > model_size - 1:
        ix = obs_density * p + obs_density - 1 + d - model_size
    else:
        ix = obs_density * p + obs_density - 1 + d

    iy = obs_density * p + obs_density - 1

    return ix, iy


@jit(nopython=True)
def addelf(nume, deno, zt, zp, model_size, nobsgrid, obs_density, obs_error_var, Pb, Hk):
    iinc_t = np.zeros((int(nobsgrid * 2), int(model_size / 2)+1))
    iinc_p = np.zeros((int(nobsgrid * 2), int(model_size / 2)+1))
    for iob in range(0, nobsgrid):
        hk = Hk[iob, :]
        hPht = hk @ Pb @ hk.T
        Pht = Pb @ hk.T
        kg = hPht / (hPht + obs_error_var)
        reg = Pht / hPht
        yt = hk @ zt
        yp = hk @ zp
        for d in range(0, model_size):
            ix, iy = covind(d, iob, obs_density, model_size)
            if d > int(model_size / 2):
                nume[model_size - d] = nume[model_size - d] + (zt[ix] - zp[ix]) * reg[ix] * kg * (yt - yp)
                deno[model_size - d] = deno[model_size - d] + reg[ix] ** 2 * kg ** 2 * (
                            (yt ** 2 + obs_error_var) - 2 * yt * yp + yp ** 2)
                iinc_t[iob + nobsgrid, model_size - d] = zt[ix] - zp[ix]
                iinc_p[iob + nobsgrid, model_size - d] = reg[ix] * kg * (yt - yp)
            else:
                nume[d] = nume[d] + (zt[ix] - zp[ix]) * reg[ix] * kg * (yt - yp)
                deno[d] = deno[d] + reg[ix] ** 2 * kg ** 2 * ((yt ** 2 + obs_error_var) - 2 * yt * yp + yp ** 2)
                iinc_t[iob, d] = zt[ix] - zp[ix]
                iinc_p[iob, d] = reg[ix] * kg * (yt - yp)
        iinc_p[nobsgrid:nobsgrid*2,0] = iinc_p[0:nobsgrid,0]
        iinc_t[nobsgrid:nobsgrid*2,0] = iinc_t[0:nobsgrid,0]
        iinc_p[nobsgrid:nobsgrid*2,-1] = iinc_p[0:nobsgrid,-1]
        iinc_t[nobsgrid:nobsgrid*2,-1] = iinc_t[0:nobsgrid,-1]

    return nume, deno, iinc_t, iinc_p


err = ensrf(ztruth, zics_total, zobs_total)
