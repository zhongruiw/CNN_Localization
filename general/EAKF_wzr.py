import numpy as np
from numba import jit
from numba import prange
from scipy.linalg import sqrtm
from numpy.linalg import inv
# from time import time
from scipy.io import savemat


def EAKF_wzr(serial_update, model_size, ensemble_size, adm_ensemble_size, nobsgrid, zens, zens_adm, Hk, obs_error_var, localize, CMat, zobs, **kwargs):
    if serial_update == 0:
        rn = 1.0 / (ensemble_size - 1)
        for iobs in range(0, nobsgrid):
            xmean = np.mean(zens, axis=0)  # 1xn
            xprime = zens - xmean
            hxens = (Hk[iobs, :] * zens.T).T  # 40*1
            hxmean = np.mean(hxens, axis=0)
            hxprime = hxens - hxmean
            hpbht = (hxprime.T * hxprime * rn)[0, 0]
            gainfact = (hpbht + obs_error_var) / hpbht * (1.0 - np.sqrt(obs_error_var / (hpbht + obs_error_var)))
            pbht = (xprime.T * hxprime) * rn

            if localize == 1:
                Cvect = CMat[iobs, :]
                eps = kwargs['eps_inde'][iobs, :]
                kfgain = np.multiply(Cvect.T, (pbht / (hpbht + obs_error_var))) + eps.T
            else:
                kfgain = pbht / (hpbht + obs_error_var)

            mean_inc = (kfgain * (zobs[0, iobs] - hxmean)).T
            prime_inc = - (gainfact * kfgain * hxprime.T).T

            zens = zens + mean_inc + prime_inc

        # zens = serial2(nobsgrid, np.array(zens), np.array(Hk), rn, obs_error_var, localize, np.array(CMat), zobs, mean_numba)
        # return np.mat(zens)
        return zens

    if serial_update == 1:
        rn = 1.0 / (ensemble_size - 1)
        for iobs in range(0, nobsgrid):
            xmean = np.mean(zens, axis=0)  # 1xn
            xprime = zens - xmean
            hxens = (Hk[iobs, :] * zens.T).T  # 40*1
            hxmean = np.mean(hxens, axis=0)
            hxprime = hxens - hxmean
            hpbht = (hxprime.T * hxprime * rn)[0, 0]
            gainfact = (hpbht + obs_error_var) / hpbht * (1.0 - np.sqrt(obs_error_var / (hpbht + obs_error_var)))
            pbht = (xprime.T * hxprime) * rn
        
            if localize == 1:
                Cvect = CMat[iobs, :]
                kfgain = np.multiply(Cvect.T, (pbht / (hpbht + obs_error_var)))
            else:
                kfgain = pbht / (hpbht + obs_error_var)

            mean_inc = (kfgain * (zobs[0, iobs] - hxmean)).T
            prime_inc = - (gainfact * kfgain * hxprime.T).T

            zens = zens + mean_inc + prime_inc

        # zens = serial2(nobsgrid, np.array(zens), np.array(Hk), rn, obs_error_var, localize, np.array(CMat), zobs, mean_numba)
        # return np.mat(zens)
        return zens

    if serial_update == 182:
        rn = 1.0 / (ensemble_size - 1)
        zpmeans = np.zeros((model_size, nobsgrid))
        phts = np.zeros((model_size, nobsgrid))
        hphts = np.zeros((1, nobsgrid))
        for iobs in range(0, nobsgrid):
            xmean = np.mean(zens, axis=0)  # 1xn
            zpmeans[:, iobs] = xmean
            xprime = zens - xmean
            hxens = (Hk[iobs, :] * zens.T).T  # 40*1
            hxmean = np.mean(hxens, axis=0)
            hxprime = hxens - hxmean
            hpbht = (hxprime.T * hxprime * rn)[0, 0]
            hphts[:,iobs] = hpbht
            gainfact = (hpbht + obs_error_var) / hpbht * (1.0 - np.sqrt(obs_error_var / (hpbht + obs_error_var)))
            pbht = (xprime.T * hxprime) * rn
            phts[:, iobs] = np.squeeze(pbht)
        
            if localize == 1:
                Cvect = CMat[iobs, :]
                kfgain = np.multiply(Cvect.T, (pbht / (hpbht + obs_error_var)))
            else:
                kfgain = pbht / (hpbht + obs_error_var)

            mean_inc = (kfgain * (zobs[0, iobs] - hxmean)).T
            prime_inc = - (gainfact * kfgain * hxprime.T).T

            zens = zens + mean_inc + prime_inc

        # zens = serial2(nobsgrid, np.array(zens), np.array(Hk), rn, obs_error_var, localize, np.array(CMat), zobs, mean_numba)
        # return np.mat(zens)
        return zens, zpmeans, phts, hphts

    # matrix assimilate time lagged observations
    if serial_update == 2:
        nobsgrid_l = int(nobsgrid*zens.shape[0])
        R  = obs_error_var * np.eye(nobsgrid_l, nobsgrid_l)
        rn = 1.0 / (ensemble_size - 1)
        HXens = np.zeros((ensemble_size, nobsgrid_l))
        for ilag in range(zens.shape[0]):   
            zens_ilag = zens[ilag, :, :]
            HXens[:, int(ilag*nobsgrid):int((ilag+1)*nobsgrid)] = (Hk @ zens_ilag.T).T  # 40*240

        HXmean = np.mean(HXens, axis=0)
        HXprime = HXens - HXmean

        Xens = zens[-1,:,:]
        Xmean = np.mean(Xens, axis=0)
        Xprime = Xens - Xmean

        CMat_l = np.array(np.tile(CMat, (zens.shape[0],1)).T)
        PHt = Xprime.T @ HXprime * rn
        HPHt = HXprime.T @ HXprime * rn
        mean_inc = np.multiply(CMat_l, PHt @ inv(HPHt + R)) @ (zobs - HXmean)

        sqrtHPHtR = sqrtm(HPHt + R).real
        pert_inc = - (np.multiply(CMat_l, PHt @ (inv(sqrtHPHtR)).T @ inv(sqrtHPHtR + sqrtm(R))) @ HXprime.T).T

        Xens = Xens + mean_inc + pert_inc

        return np.mat(Xens)

    elif serial_update == 3:
        rn = 1.0 / (ensemble_size - 1)
        lamda = 1 - theta
        zens_prior = zens

        for iobs in range(0, nobsgrid):
            xmean = np.mean(zens, axis=0)  # 1xn
            xprime = zens - xmean
            hxens = (Hk[iobs, :] * zens.T).T  # 40*1
            hxmean = np.mean(hxens, axis=0)
            hxprime = hxens - hxmean
            hpbht = (hxprime.T * hxprime * rn)[0, 0]
            gainfact = (hpbht + obs_error_var) / hpbht * (1.0 - np.sqrt(obs_error_var / (hpbht + obs_error_var)))
            pbht = (xprime.T * hxprime) * rn
        
            if localize == 1:
                Cvect = CMat[iobs, :]
                kfgain = np.multiply(Cvect.T, (pbht / (hpbht + obs_error_var)))
            else:
                kfgain = pbht / (hpbht + obs_error_var)
        
            xmean = xmean + (kfgain * (zobs[0, iobs] - hxmean)).T  # 1 x n
            xprime = xprime - (gainfact * kfgain * hxprime.T).T  # ens x n
            zens = xprime + xmean

        # zens = np.mat(serial2(nobsgrid, np.array(zens), np.array(Hk), rn, obs_error_var, localize, np.array(CMat), zobs, mean_numba))

        obs_inc_en_mean = np.mean(zens, axis=0) - np.mean(zens_prior, axis=0)
        obs_inc_en_prime = zens - zens_prior - obs_inc_en_mean

        HXens = (Hk * zens_prior.T).T  # 40*240
        HXmean = np.mean(HXens, axis=0)
        obs_inc_3dvar = B * Hk.T * (Hk*B*Hk.T + R).I * (zobs.T - HXmean.T)  # ensemble mean as 3dvar background

        # ---------------------  hybrid 3dvar increments ----------------------
        zens = zens_prior + lamda * obs_inc_en_mean + theta * obs_inc_3dvar.T + obs_inc_en_prime
        # ---------------------------------------------------------------------

        return zens


@jit(nopython=True)
def serial2(nobsgrid, zens, Hk, rn, obs_error_var, localize, CMat, zobs, mean_numba):
    for iobs in range(0, nobsgrid):
        xmean = mean_numba(zens)  # 1xn
        xprime = zens - xmean
        hxens = (Hk[iobs, :] @ zens.T).T  # 40*1
        hxmean = np.mean(hxens)
        hxprime = hxens - hxmean
        hpbht = (hxprime.T @ hxprime * rn)
        gainfact = (hpbht + obs_error_var) / hpbht * (1.0 - np.sqrt(obs_error_var / (hpbht + obs_error_var)))
        pbht = (xprime.T @ hxprime) * rn

        if localize == 1:
            Cvect = CMat[iobs, :]
            kfgain = Cvect * (pbht/(hpbht+obs_error_var))
        else:
            kfgain = pbht / (hpbht + obs_error_var)

        xmean = xmean + (kfgain * (zobs[0, iobs] - hxmean)).T  # 1 x n
        xprime = xprime - (gainfact * kfgain.reshape(kfgain.shape[0], 1) @ hxprime.reshape(1, hxprime.shape[0])).T  # ens x n

        zens = xprime + xmean
    return zens


@jit(parallel=True)
def mean_numba(a):
    res = []
    for i in prange(a.shape[1]):
        res.append(a[:, i].mean())

    return np.array(res)

