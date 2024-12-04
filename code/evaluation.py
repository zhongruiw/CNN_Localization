# ----- evalution -----
# version: 1.0

import numpy as np
from construct_GC_2d import construct_GC_2d
from construct_func_2d import construct_func_2d
from EnKF import EnKF
from os.path import dirname, join as pjoin
from scipy.io import loadmat
from step_L04 import step_L04
import torch
from basicsr.models import create_model
from basicsr.utils.options import parse


np.random.seed(2022)


# load data
data_dir = '../data/assimilation'
icsfname = pjoin(data_dir, 'ics_ms3_from_zt1year_sz3001.mat')
obsfname = pjoin(data_dir, 'obs_ms3_err1_240s_6h_25y.mat')
truthfname = pjoin(data_dir, 'zt_25year_ms3_6h.mat')

# get truth
time_steps = 200 * 450  # days(1 day~ 200 time steps)
obs_freq_timestep = 50
obsdens = 4
truthf = loadmat(truthfname)
eva_bg = int(23 * 360 * 200 / obs_freq_timestep) # choose an different period from training for independent evaluation
eva_ed = eva_bg+int(time_steps/obs_freq_timestep+1)
ztruth = truthf['zens_times'][eva_bg:eva_ed, :]

# get initial condition
icsf = loadmat(icsfname)
zics_total = icsf['zics_total1']

# get observation
obsf = loadmat(obsfname)
zobs_total = obsf['zobs_total'][eva_bg:eva_ed, :]

# analysis period
iobsbeg = 361
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
ens_mem_beg = 248 # different initials for evaluation
ens_mem_end = ens_mem_beg + ensemble_size

inflation_value = 1.01

# choose localization scheme
    # clf: CNN-based localization funciton
    # celf: CNN-based emperical localization funciton
    # elf: emperical localization funciton
    # lrlf: linear regression localization funciton
    # gc: Gaspari & Cohn localization funciton
localize = 1
loc = 'clf'
if loc = 'clf':
    opt_path_denoise = "/clf/options/GainMat/test/SRCNN-F15-sob-64-enkf.yml"
    opt_denoise = parse(opt_path_denoise, is_train=False)
    opt_denoise["dist"] = False
    model = create_model(opt_denoise)
elif loc = 'celf':
    # CNN-based localization funciton
    opt_path_denoise = "/clf/options/GainMat/test/SRCNN-F15-sob-64-enkf-xtloss.yml"
    opt_denoise = parse(opt_path_denoise, is_train=False)
    opt_denoise["dist"] = False
    model = create_model(opt_denoise)
elif loc = 'elf':
    locf = np.load('/train/elf/iter3/elf_smsp.npy')
    localize_func = np.concatenate((locf, locf[-2:0:-1]), axis=0)
    CMat = np.mat(construct_func_2d(localize_func, model_size, obs_grids))
elif loc = 'lrlf':
    locf = np.load('/train/lrlf/lrlf_smsp.npy')
    localize_func = np.concatenate((locf, locf[-2:0:-1]), axis=0)
    CMat = np.mat(construct_func_2d(localize_func, model_size, obs_grids))
elif loc = 'gc':
    localization_value = 240
    CMat = np.mat(construct_GC_2d(localization_value, model_size, obs_grids))

zens = np.mat(zics_total[ens_mem_beg: ens_mem_end, :])  # ensemble are drawn from ics set
zeakf_prior = np.zeros((model_size, nobstime))
zeakf_analy = np.empty((model_size, nobstime))

for iassim in range(0, nobstime):
    print(iassim)
    # EnKF step
    obsstep = iassim * obs_freq_timestep + 1

    zeakf_prior[:, iassim] = np.mean(zens, axis=0)  # prior ensemble mean
    prior_spread[:, iassim] = np.std(zens, axis=0, ddof=1)

    # perturn observations
    zobs = np.mat(zobs_total[iassim, :])
    obs_p = np.random.normal(0., np.sqrt(obs_error_var), (ensemble_size,nobsgrid))
    obs = zobs + obs_p
    
    # inflation
    ensmean = np.mean(zens, axis=0)
    ensp = zens - ensmean
    zens = ensmean + ensp * inflation_value

    # save inflated prior zens
    zens_prior = zens

    rn = 1.0 / (ensemble_size - 1)
    Xmean = np.mean(zens, axis=0)
    Xprime = zens - Xmean
    HXens = (Hk * zens.T).T
    HXprime = HXens - (Hk * Xmean.T).T
    PbHt = (Xprime.T * HXprime) * rn
    HPbHt = (HXprime.T * HXprime) * rn
    kg = PbHt * (HPbHt + R).I

    if loc == 'clf' or loc == 'celf':    
        # nn predict
        inp = np.array(kg)[:,:,None]
        inp = torch.from_numpy(inp.transpose(2, 0, 1))
        inp = inp.float()
        model.feed_data(data={"lq": inp.unsqueeze(dim=0)})
        model.test()
        visuals = model.get_current_visuals()
        kg_ai = np.mat(visuals['result'].numpy().squeeze())

        # matrix EnKF undate
        HXens = Hk * zens_prior.T
        obs_inc = kg_ai * (obs.T - HXens)
        zens = zens_prior + obs_inc.T

    if loc == 'elf' or 'lrlf' or 'gc':
        # serial EnKF update
        zens = EnKF(model_size, ensemble_size,
                        nobsgrid, zens, Hk, obs_error_var, localize, CMat, obs)

    # save
    zens_analy = np.mean(zens, axis=0)
    zeakf_analy[:, iassim] = zens_analy

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
        
zt = ztruth.T

prior_rmse = np.sqrt(np.mean((zt - zeakf_prior) ** 2, axis=0))
analy_rmse = np.sqrt(np.mean((zt - zeakf_analy) ** 2, axis=0))
prior_err = np.mean(prior_rmse[iobsbeg - 1: iobsend])
analy_err = np.mean(analy_rmse[iobsbeg - 1: iobsend])

print('prior error = {0:.6f}'.format(prior_err))
print('analysis error = {0:.6f}'.format(analy_indeself_err))

# save
np.save('prior_rmse.npy', prior_rmse)
np.save('analy_rmse.npy', analy_rmse)
