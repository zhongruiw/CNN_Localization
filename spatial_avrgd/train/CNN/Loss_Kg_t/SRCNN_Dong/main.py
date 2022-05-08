# ----- model training -----
# coding by Zhongrui Wang
# version: 2.3
# update: average Hk, add kg rolling, fix random seed

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.layers import InputSpec
from tensorflow.python.keras.utils import conv_utils
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from give_a_model import srcnn
from scipy.io import loadmat


# to limit tensorflow to a specific sets of gpus
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_visible_devices(gpus[0:2], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

# fix random seed
tf.random.set_seed(1234)

# ------------------ experiment settings ------------------
model_size = 960
obs_dens = 4
nobs = int(model_size / obs_dens)
model_grids = np.arange(1, model_size + 1)
obs_grids = model_grids[model_grids % obs_dens == 0]
nobsgrid = nobs

# make forward operator H
Hk = np.mat(np.zeros((nobs, model_size)))
# for iobs in range(0, nobs):
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

# -------------------- prepare data ----------------------
# load data
kg_f = np.load('/scratch/lllei/spatially_averaged/200040/avrgd_025/obs_dens4/kg_f_5y.npy')
# kg_f = np.load('/scratch/lllei/kg_f_25y_0.npy')
# for i in range(1,5):
#     fname = '/scratch/lllei/kg_f_25y_{:d}.npy'
#     kg_f = np.concatenate((kg_f, np.load(fname.format(i))), axis=0)

# Loss: kgain mse
kg_t = np.load('/scratch/lllei/spatially_averaged/200040/avrgd_025/obs_dens4/kg_t_5y.npy')

# roll kg (optional)
# for j in range(0, nobs):
#     kg_f[:, :, j] = np.roll(kg_f[:, :, j], -(obs_dens*j+obs_dens-1), axis=1)
#     kg_t[:, :, j] = np.roll(kg_t[:, :, j], -(obs_dens*j+obs_dens-1), axis=1)

# exclude the unsaturated period
cut_out = 50
x = kg_f[cut_out:, :, :, np.newaxis]
y = kg_t[cut_out:, :, :, np.newaxis]

## Loss: xt
# truthf = loadmat('/home/lllei/data/zt_5year_ms3_6h.mat')
# ztruth = truthf['zens_times']
# xt = ztruth[1:,:]
# xf = np.load('/home/lllei/AI_localization/L05/e2000_inf1050_e40_inf1050_loc240/200040/save_interval_6h/zeakf_prior.npy')
# xf = xf[:, 1:]
# obs = np.load('/home/lllei/AI_localization/L05/e2000_inf1050_e40_inf1050_loc240/200040/save_interval_6h/observations.npy')
# obs = obs[1:, :]
# incs = obs - (Hk @ xf).T
# xf = xf.T
# xt_xf_inc = np.concatenate((xt,xf,incs), axis=1)

# if np.shape(xt_xf_inc)[0] != np.shape(kg_f)[0]:
#     raise ValueError('check the consistence between xt and Kgain_f')

# # exclude the unsaturated period
# cut_out = 50
# x = kg_f[cut_out:, :, :, np.newaxis]
# y = xt_xf_inc[cut_out:, :, np.newaxis]

# split dataset into test set and train-val dataset
x_traval, x_test, y_traval, y_test = train_test_split(x, y, test_size=0.1, random_state=248)

# turn np.array data into tf.data.Dataset
batch_size = 8
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(batch_size)

# --------------------- specify a model ---------------------
model = srcnn(batch_size=batch_size)
model.summary()
model.output_shape

kfold_train = False

# ----------------------- training --------------------------
if kfold_train == True:
    # Define per-fold score containers
    loss_per_fold = []
    metric_per_fold = []
    # Define the K-fold Cross Validator
    num_folds = 9
    kfold = KFold(n_splits=num_folds, shuffle=False)

    # K-fold Cross Validation model evaluation
    fold_no = 1
    for itrain, ival in kfold.split(kg_f_traval, kg_t_traval):
        kg_f_train = kg_f_traval[itrain, :, :, :]
        kg_t_train = kg_t_traval[itrain, :, :, :]
        kg_f_val = kg_f_traval[ival, :, :, :]
        kg_t_val = kg_t_traval[ival, :, :, :]
        train_dataset = tf.data.Dataset.from_tensor_slices((kg_f_train, kg_t_train))
        val_dataset = tf.data.Dataset.from_tensor_slices((kg_f_val, kg_t_val))

        # shuffle and batch the datasets
        train_dataset = train_dataset.shuffle(buffer_size=kg_f_train.shape[0], reshuffle_each_iteration=True).batch(batch_size)
        val_dataset = val_dataset.shuffle(buffer_size=kg_f_val.shape[0], reshuffle_each_iteration=True).batch(batch_size)

        # define and compile the model
        model = srcnn()

        # Generate a print
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')

        # train the model
        history = model.fit(train_dataset, epochs=15, validation_data=val_dataset)
        keys = list(history.history.keys())
        print(
            f'fold {fold_no}: {keys[0]} : {history.history[keys[0]][-1]:.4e} -  '
            f'{keys[1]} : {history.history[keys[1]][-1]:.4e} - '
            f'{keys[2]} : {history.history[keys[2]][-1]:.4e} - '
            f'{keys[3]} : {history.history[keys[3]][-1]:.4e}')

        loss_per_fold.append(history.history[keys[2]][-1])
        metric_per_fold.append(history.history[keys[3]][-1])

        # evaluate the model
        print(f'Evaluating for fold {fold_no} ...')
        result = model.evaluate(test_dataset)
        dict(zip(model.metrics_names, result))

        # model.save('/home/lllei/AI_localization/L05/server/inf1050_loc240/200040/model_0layer')
        save_dir = './fold{0:d}/my_weights/srcnn'
        model.save_weights(save_dir.format(fold_no))

        figure, axis = plt.subplots(2, 1, sharex=True)
        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        axis[0].plot(history.history['loss'])
        axis[0].plot(history.history['val_loss'])
        # axis[0,0].title('model')
        axis[0].set_ylabel('loss')
        axis[0].legend(['train', 'validation'], loc='upper right')

        axis[1].plot(history.history['gradient_norm'])
        axis[1].plot(history.history['val_gradient_norm'])
        # axis[0,0].title('model')
        axis[1].set_ylabel('gradient norm')
        axis[1].set_xlabel('epoch')
        axis[1].legend(['train', 'validation'], loc='upper right')
        plt.show()
        save_dir = './fold{0:d}/training_history.png'
        plt.savefig(save_dir.format(fold_no))

        # Increase fold number
        fold_no = fold_no + 1

    # == Provide average scores ==
    print('------------------------------------------------------------------------')
    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(num_folds):
        print(f'> Fold {i + 1} - {keys[2]}: {loss_per_fold[i]:.4e} - {keys[3]}: {metric_per_fold[i]:.4e}')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Loss: {np.mean(loss_per_fold):.4e}')
    print(f'> {keys[1]}: {np.mean(metric_per_fold):.4e} (+- {np.std(metric_per_fold):.4e})')
    print('------------------------------------------------------------------------')

else:
    x_train, x_val, y_train, y_val = train_test_split(x_traval, y_traval, test_size=0.1, random_state=248)

    # turn np.array data into tf.data.Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    # shuffle and batch the datasets
    train_dataset = train_dataset.shuffle(buffer_size=x_train.shape[0], reshuffle_each_iteration=True).batch(batch_size)
    val_dataset = val_dataset.shuffle(buffer_size=y_val.shape[0], reshuffle_each_iteration=True).batch(batch_size)

    # define and compile the model
    model = srcnn(batch_size=batch_size)

    # train the model
    history = model.fit(train_dataset, epochs=4, validation_data=val_dataset)
    keys = list(history.history.keys())

    # evaluate the model       
    print('Evaluating ...')
    result = model.evaluate(test_dataset)
    dict(zip(model.metrics_names, result))

    # model.save('/home/lllei/AI_localization/L05/server/inf1050_loc240/200040/model_0layer')
    save_dir = './my_weights/srcnn'
    model.save_weights(save_dir)

    figure, axis = plt.subplots(2, 1, sharex=True)
    axis[0].plot(history.history['loss'])
    axis[0].plot(history.history['val_loss'])
    # axis[0,0].title('model')
    axis[0].set_ylabel('loss')
    axis[0].legend(['train', 'validation'], loc='upper right')

    axis[1].plot(history.history['gradient_norm'])
    axis[1].plot(history.history['val_gradient_norm'])
    # axis[0,0].title('model')
    axis[1].set_ylabel('metrics')
    axis[1].set_xlabel('epoch')
    axis[1].legend(['train', 'validation'], loc='upper right')
    plt.show()
    save_dir = './training_history.png'
    plt.savefig(save_dir)

    np.save('training_history.npy', history.history)
