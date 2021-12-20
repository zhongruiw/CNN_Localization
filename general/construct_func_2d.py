import numpy as np
# from numba import jit


# @jit(nopython=True)
def construct_func_2d(loc_f, model_size, obs_grids):
    nobs = len(obs_grids)
    obs_dens = int(model_size / nobs)
    V = np.zeros((nobs, model_size))

    for iobs in range(0, nobs):
        V[iobs, :] = np.roll(loc_f, int(obs_dens*iobs)+obs_dens-1)

    return V
