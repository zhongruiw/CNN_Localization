import numpy as np


def H(opt, model_size, obs_dens, **kwargs):
    model_grids = np.arange(1, model_size + 1)
    obs_grids = model_grids[model_grids % obs_dens == 0]
    nobsgrid = len(obs_grids)

    nobsgrid = len(obs_grids)
    Hk = np.mat(np.zeros((nobsgrid, model_size)))

    if opt == 'single':
        for iobs in range(0, nobsgrid):
            x1 = obs_grids[iobs] - 1
            Hk[iobs, x1] = 1.0

        return Hk

    elif opt == 'average':
        ave_range = kwargs['ave_range'] * model_size
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

        return Hk

    elif opt == 'gamma_weighted':
        gamma = kwargs['gamma']
        theta = kwargs['theta']
        for iobs in range(nobsgrid):
            Hk[iobs,:] = np.roll(gamma, -theta+(iobs+1)*obs_dens-1)

        return Hk

    else:
        raise ValueError('wrong option, current options are: \'single\', \'average\', \'gamma_weighted\'')



if __name__ == '__main__':
    # model parameters
    model_size = 960  # N
    # observation parameters
    obs_dens = 4

    gamma = np.load('gamma_k2_theta40.npy')
    theta = 40
    Hk = H('gamma_weighted', model_size, obs_dens, gamma=gamma, theta=theta)
#     Hk = H('single', model_size, obs_dens)
#     Hk = H('average', model_size, obs_dens,ave_range=0.25)

    print(Hk[0,:])
