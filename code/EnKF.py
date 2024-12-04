# stochastic EnKF
import numpy as np


def EnKF(model_size, ensemble_size, nobsgrid, zens, Hk, obs_error_var, localize, CMat, zobs, **kwargs):
    rn = 1.0 / (ensemble_size - 1)

    for iobs in range(0, nobsgrid):
        xmean = np.mean(zens, axis=0)  # 1xn
        xprime = zens - xmean
        hxens = (Hk[iobs, :] * zens.T).T  # 40*1
        hxmean = np.mean(hxens, axis=0)
        hxprime = hxens - hxmean
        hpbht = (hxprime.T * hxprime * rn)[0, 0]
        pbht = (xprime.T * hxprime) * rn
    
        if localize == 1:
            Cvect = CMat[iobs, :]
            kfgain = np.multiply(Cvect.T, (pbht / (hpbht + obs_error_var)))
        else:
            kfgain = pbht / (hpbht + obs_error_var)

        inc = (kfgain * (zobs[:,iobs] - hxens).T).T

        zens = zens + inc

    return zens
