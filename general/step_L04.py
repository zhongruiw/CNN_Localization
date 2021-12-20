# Runge-Kutta scheme
import numpy as np
# from time import time
from numba import jit
# from numba import cuda
# from numba import guvectorize, float64


#@guvectorize(['void(int64, float64[:,:], int64, int64, int64, float64[:], int64, int64,int64,int64, float64[:,:])'],'(m,n)->(m,n)')
def step_L04(ensemble_size, zens, model_size, ss2, smooth_steps, a, model_number, K4, H, K, K2, sts2, coupling, space_time_scale, forcing, delta_t):
    for iens in range(0, ensemble_size):
        z = zens[iens, :].T
        
        z_save = z
        dz = comp_dt_L04(z, model_size, ss2, smooth_steps, a, model_number, K4, H, K, K2, sts2, coupling, space_time_scale, forcing)  # Compute the first intermediate step
        z1 = np.multiply(delta_t, dz)
        z = z_save + z1 / 2.0

        dz = comp_dt_L04(z, model_size, ss2, smooth_steps, a, model_number, K4, H, K, K2, sts2, coupling, space_time_scale, forcing)  # Compute the second intermediate step
        z2 = np.multiply(delta_t, dz)
        z = z_save + z2 / 2.0

        dz = comp_dt_L04(z, model_size, ss2, smooth_steps, a, model_number, K4, H, K, K2, sts2, coupling, space_time_scale, forcing)  # Compute the third intermediate step
        z3 = np.multiply(delta_t, dz)
        z = z_save + z3

        dz = comp_dt_L04(z, model_size, ss2, smooth_steps, a, model_number, K4, H, K, K2, sts2, coupling, space_time_scale, forcing)  # Compute fourth intermediate step
        z4 = np.multiply(delta_t, dz)

        dzt = z1 / 6.0 + z2 / 3.0 + z3 / 3.0 + z4 / 6.0
        z = z_save + dzt
        
        zens[iens, :] = z.T

    # gpu_version    
    #z_device = cuda.to_device(np.array(z))
    #a_device = cuda.to_device(a)
    # set block & grid dimension
    #blockdim = model_size
    #griddim = ensemble_size

    #gpu_step_z[griddim, blockdim](delta_t, zens, model_size, ss2, smooth_steps, a, model_number, K4, H, K, K2, sts2, coupling, space_time_scale, forcing)
    #cuda.synchronize()
    #zens_gpu = zens.copy_to_host()
    
    return zens


# @cuda.jit
# def gpu_step_z(delta_t, zens, model_size, ss2, smooth_steps, a, model_number, K4, H, K, K2, sts2, coupling, space_time_scale, forcing):
#     iens = cuda.blockIdx.x
#     #i = cuda.threadIdx.x
#     z = zens[iens, :].T

#     z_save = z
#     dz = gpu_comp_dt_L04(z, model_size, ss2, smooth_steps, a, model_number, K4, H, K, K2, sts2, coupling, space_time_scale, forcing)  # Compute the first intermediate step
#     z1 = np.multiply(delta_t, dz)
#     z = z_save + z1 / 2.0

#     dz = gpu_comp_dt_L04(z, model_size, ss2, smooth_steps, a, model_number, K4, H, K, K2, sts2, coupling, space_time_scale, forcing)  # Compute the second intermediate step
#     z2 = np.multiply(delta_t, dz)
#     z = z_save + z2 / 2.0

#     dz = gpu_comp_dt_L04(z, model_size, ss2, smooth_steps, a, model_number, K4, H, K, K2, sts2, coupling, space_time_scale, forcing)  # Compute the third intermediate step
#     z3 = np.multiply(delta_t, dz)
#     z = z_save + z3

#     dz = gpu_comp_dt_L04(z, model_size, ss2, smooth_steps, a, model_number, K4, H, K, K2, sts2, coupling, space_time_scale, forcing)  # Compute fourth intermediate step
#     z4 = np.multiply(delta_t, dz)

#     dzt = z1 / 6.0 + z2 / 3.0 + z3 / 3.0 + z4 / 6.0
#     z = z_save + dzt
    
#     zens[iens, :] = z.T



def comp_dt_L04(z, model_size, ss2, smooth_steps, a, model_number, K4, H, K, K2, sts2, coupling, space_time_scale, forcing):
    if model_number == 3:
        x, y = z2xy(z, model_size, ss2, smooth_steps, a)
    elif model_number == 2:
        x = z
        y = 0.0 * z
    else:
        print('Do not know that model number')
        return

    #  Deal with  # cyclic boundary# conditions using buffers

    # Fill the xwrap and ywrap buffers
    xwrap = np.concatenate([x[model_size-K4: model_size], x, x[0: K4]])
    ywrap = np.concatenate([y[model_size-K4: model_size], y, y[0: K4]])

    wx = np.mat(np.zeros((model_size+K4*2, 1)))
    # ! Calculate the W's
    wx = calw(K, H, K4, model_size, wx, xwrap)

    # Fill the W buffers
    wx[0: K4, 0] = wx[model_size: model_size + K4, 0]
    wx[model_size+K4: model_size+2*K4, 0] = wx[K4: K4*2, 0]

    dz = np.mat(np.zeros((model_size, 1)))
    # ! Generate dz / dt
    dz = caldz(K, H, K4, K2, model_size, wx, xwrap, model_number, dz, sts2, ywrap, coupling, space_time_scale, forcing)

    return dz


# @cuda.jit(device=True)
# def gpu_comp_dt_L04(z, model_size, ss2, smooth_steps, a, model_number, K4, H, K, K2, sts2, coupling, space_time_scale, forcing):
#     if model_number == 3:
#         x, y = gpu_z2xy(z, model_size, ss2, smooth_steps, a)
#     elif model_number == 2:
#         x = z
#         y = 0.0 * z
#     else:
#         print('Do not know that model number')
#         return

#     #  Deal with  # cyclic boundary# conditions using buffers

#     # Fill the xwrap and ywrap buffers
#     xwrap = np.concatenate([x[model_size-K4: model_size], x, x[0: K4]])
#     ywrap = np.concatenate([y[model_size-K4: model_size], y, y[0: K4]])

#     wx = np.mat(np.zeros((model_size+K4*2, 1)))
#     # ! Calculate the W's
#     wx = calw(K, H, K4, model_size, wx, xwrap)

#     # Fill the W buffers
#     wx[0: K4, 0] = wx[model_size: model_size + K4, 0]
#     wx[model_size+K4: model_size+2*K4, 0] = wx[K4: K4*2, 0]

#     dz = np.mat(np.zeros((model_size, 1)))
#     # ! Generate dz / dt
#     dz = caldz(K, H, K4, K2, model_size, wx, xwrap, model_number, dz, sts2, ywrap, coupling, space_time_scale, forcing)

#     return dz


def z2xy(z, model_size, ss2, smooth_steps, a):
    # ss2 is smoothing scale I * 2
    # Fill zwrap
    zwrap = np.concatenate([z[(model_size - ss2 - 1): model_size], z, z[0: ss2]])
    x = np.mat(np.zeros((model_size, 1)))
    y = np.mat(np.zeros((model_size, 1)))

    # CPU version -- Generate the x variables
    #start_t2 = time()
    x = calx(x, ss2, model_size, a, zwrap, smooth_steps)
    #print('cpu_calx_time:' + str(time()-start_t2))

    # GPU version -- Generate the x variables
    #start_t1 = time()
    # transfer data from host to device
    #x_device = cuda.to_device(np.array(x))
    #a_device = cuda.to_device(a)
    #zwrap_device = cuda.to_device(np.array(zwrap))
    # set block & grid dimension
    #blockdim = model_size
    #griddim = 1

    #gpu_calx[griddim, blockdim](x_device, ss2, model_size, a_device, zwrap_device, smooth_steps)
    #cuda.synchronize()
    #x_gpu = x_device.copy_to_host()

    #print('gpu_calx_time:' + str(time()-start_t1))

    #if (np.array_equal(x, x_gpu)):
    #    print('result correct.')
    #else:
    #    print(x[0:3]-x_gpu[0:3])

    # Generate the y variables
    y = z - x
    return x, y

# @cuda.jit(device=True)
# def gpu_z2xy(z, model_size, ss2, smooth_steps, a):
#     # ss2 is smoothing scale I * 2
#     # Fill zwrap
#     zwrap = np.concatenate([z[(model_size - ss2 - 1): model_size], z, z[0: ss2]])
#     x = np.mat(np.zeros((model_size, 1)))
#     y = np.mat(np.zeros((model_size, 1)))

#     # transfer data from host to device
#     x_device = cuda.to_device(np.array(x))
#     a_device = cuda.to_device(a)
#     zwrap_device = cuda.to_device(np.array(zwrap))

#     gpu_calx(x_device, ss2, model_size, a_device, zwrap_device, smooth_steps)
#     cuda.synchronize()
#     x_gpu = x_device.copy_to_host()

#     # Generate the y variables
#     y = z - x


# CPU version
@jit(nopython=True)
def calx(x, ss2, model_size, a, zwrap, smooth_steps):
    for i in range(ss2, ss2 + model_size):
        x[i - ss2, 0] = a[0] * zwrap[i + 1 - (- smooth_steps), 0] / 2.00
        for j in range(- smooth_steps + 1, smooth_steps):
            x[i - ss2, 0] = x[i - ss2, 0] + a[j + smooth_steps] * zwrap[i + 1 - j, 0]
        x[i - ss2, 0] = x[i - ss2, 0] + a[2 * smooth_steps] * zwrap[i + 1 - smooth_steps, 0] / 2.00
    return x


# GPU version
# @cuda.jit(device=True)
# def gpu_calx(x_device, ss2, model_size, a_device, zwrap_device, smooth_steps):
#     # x = cuda.shared.array(shape=(model_size, 1), dtype=float64)
#     # zwap = cuda.shared.array(shape=(model_size, 1), dtype=float64)
#     a = a_device
#     zwrap = zwrap_device
#     i = cuda.threadIdx.x + ss2
#     # j = cuda.threadIdx.x - smooth_steps

#     x_device[i - ss2, 0] = a[0] * zwrap[i + 1 - (- smooth_steps), 0] / 2.00
#     for j in range(- smooth_steps + 1, smooth_steps):
#         x_device[i - ss2, 0] = x_device[i - ss2, 0] + a[j + smooth_steps] * zwrap[i + 1 - j, 0]
#     x_device[i - ss2, 0] = x_device[i - ss2, 0] + a[2 * smooth_steps] * zwrap[i + 1 - smooth_steps, 0] / 2.00
#     #if j == (- smooth_steps):
#     #    x_device[i - ss2, 0] = x_device[i - ss2, 0] + a[0] * zwrap[i + 1 - (- smooth_steps), 0] / 2.00
#     #elif j == smooth_steps:
#     #    x_device[i - ss2, 0] = x_device[i - ss2, 0] + a[2 * smooth_steps] * zwrap[i + 1 - smooth_steps, 0] / 2.00
#     #else:
#     #    x_device[i - ss2, 0] = x_device[i - ss2, 0] + a[j + smooth_steps] * zwrap[i + 1 - j, 0]

    

@jit(nopython=True)
def calw(K, H, K4, model_size, wx, xwrap):
    # ! Calculate the W's
    for i in range(K4, K4 + model_size):
        wx[i, 0] = xwrap[i - (-H), 0] / 2.00
        for j in range(- H + 1, H):
            wx[i, 0] = wx[i, 0] + xwrap[i - j, 0]

        wx[i, 0] = wx[i, 0] + xwrap[i - H, 0] / 2.00
        wx[i, 0] = wx[i, 0] / K
    return wx


@jit(nopython=True)
def caldz(K, H, K4, K2, model_size, wx, xwrap, model_number, dz, sts2, ywrap, coupling, space_time_scale, forcing):
    for i in range(K4, K4 + model_size):
        xx = wx[i - K + (-H), 0] * xwrap[i + K + (-H), 0] / 2.00
        for j in range(- H + 1, H):
            xx = xx + wx[i - K + j, 0] * xwrap[i + K + j, 0]
        xx = xx + wx[i - K + H, 0] * xwrap[i + K + H, 0] / 2.00
        xx = - wx[i - K2, 0] * wx[i - K, 0] + xx / K

        if model_number == 3:
            dz[i - K4, 0] = xx + sts2 * (- ywrap[i - 2, 0] * ywrap[i - 1, 0] + ywrap[i - 1, 0] * ywrap[i + 1, 0])\
                            + coupling * (- ywrap[i - 2, 0] * xwrap[i - 1, 0] + ywrap[i - 1, 0] * xwrap[i + 1, 0]) - xwrap[i, 0]\
                            - space_time_scale * ywrap[i, 0] + forcing
        else:  # must be model II
            dz[i - K4, 0] = xx - xwrap[i, 0] + forcing

    return dz
