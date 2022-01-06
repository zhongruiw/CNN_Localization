# ----- srcnn_dnn_xt model -----
# coding by Zhongrui Wang
# version: 1.0

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.layers import InputSpec
from tensorflow.python.keras.utils import conv_utils


def srcnn(**kwargs):
    # -------------------- custom training --------------------
    class PeriodicPadding2D(layers.Layer):
        def __init__(self, padding=(1, 1), **kwargs):
            super(PeriodicPadding2D, self).__init__(**kwargs)
            self.padding = conv_utils.normalize_tuple(padding, 2, 'padding')
            self.input_spec = InputSpec(ndim=4)

        def wrap_pad(self, input, size_row, size_col):
            M1 = tf.concat([input[:, :, -size_col:, :], input, input[:, :, 0:size_col, :]], 2)
            M1 = tf.concat([M1[:, -size_row:, :, :], M1, M1[:, 0:size_row, :, :]], 1)
            return M1

        def compute_output_shape(self, input_shape):
            shape = list(input_shape)
            assert len(shape) == 4
            if shape[1] is not None:
                length_0 = shape[1] + 2 * self.padding[0]
                length_1 = shape[2] + 2 * self.padding[1]
            else:
                length = None
            return tuple([shape[0], length_0, length_1, shape[3]])

        def call(self, inputs):
            return self.wrap_pad(inputs, self.padding[0], self.padding[1])

        def get_config(self):
            config = {'padding': self.padding}
            base_config = super(PeriodicPadding2D, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    class Loc_by_1D(layers.Layer):
        def __init__(self, obs_dens, model_size, nobs, batch_size, **kwargs):
            super(Loc_by_1D, self).__init__(**kwargs)
            self.obs_dens = obs_dens
            self.model_size = model_size
            self.nobs = nobs
            self.batch_size = batch_size

        def rotate(self, matrix, shifts):
            """"assumes matrix shape is bxmxn (batch_size b)and shifts dimension is m"""

            # get shape of the input matrix
            shape = tf.shape(matrix)

            # compute and stack the meshgrid to get the index matrix of shape (2,m,n)
            ind = tf.stack(tf.meshgrid(tf.range(shape[1]), tf.range(shape[2]), indexing='ij'))
            # reshape it to (m,n,2)
            ind = tf.transpose(ind, [1, 2, 0])

            # add the value from shifts to the corresponding row and devide modulo shape[1]
            # this will effectively introduce the desired shift, but at the level of indices
            shifted_ind = tf.math.floormod(ind[:, :, 0] + shifts, shape[1])

            # convert the shifted indices to the right shape
            new_ind = tf.transpose(tf.stack([shifted_ind, ind[:, :, 1]]), [1, 2, 0])

            # repeat new_ind batch_size times
            b = tf.concat([[shape[0]], [1, 1, 1]], axis=0)
            new_ind = tf.tile(tf.expand_dims(new_ind, axis=0), b)

            # return the resliced tensor
            return tf.gather_nd(matrix, new_ind, batch_dims=1)

        def call(self, inputs):
            kg_f = tf.squeeze(inputs[0])
            loc_f = tf.squeeze(inputs[1])

            b = tf.constant([1, 1, self.nobs], tf.int32)
            loc_mat = tf.tile(tf.expand_dims(tf.concat([loc_f, loc_f[:, -2:0:-1]], axis=1), axis=-1), b)
            shifts = tf.cast(-tf.linspace(self.obs_dens-1.0, self.model_size-1.0, self.nobs), tf.int32)

            loc_mat_shift = self.rotate(loc_mat, shifts)

            kg_pred = tf.math.multiply(kg_f, loc_mat_shift)
            return kg_pred

        def compute_output_shape(self, input_shape):
            shape = list(input_shape)
            assert len(shape) == 2
            return tuple([shape[0], self.model_size, self.nobs])

        def get_config(self):
            config = {'obs_dens': self.obs_dens, 'model_size': self.model_size,
                      'nobs': self.nobs, 'batch_size': self.batch_size}
            base_config = super(Loc_by_1D, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    class GradientNorm(tf.keras.metrics.Metric):
        def __init__(self, name='gradient_norm', **kwargs):
            super().__init__(name=name, **kwargs)
            self.gd_norm = self.add_weight(name='gdnorm', initializer='zeros')

        def update_state(self, y, gradients, sample_weight=None):
            # self.gd_norm.assign(tf.norm(gradients, ord='euclidean'))
            norm = tf.norm(gradients, ord='euclidean') / tf.cast(tf.size(gradients, out_type=tf.int32), tf.float32)
            self.gd_norm.assign(norm)

        def result(self):
            return self.gd_norm

        def reset_states(self):
            self.gd_norm.assign(0)

    class LocLoss(keras.losses.Loss):
        def __init__(self, kg_raw, model_size, nobs, batch_size, name="custom_loss"):
            super().__init__(name=name)
            self.kg_raw = kg_raw
            self.model_size = model_size
            self.nobs = nobs
            self.batch_size = batch_size

        def call(self, kg_true, loc_f):
            k = tf.constant([1, 1, self.nobs, 1], tf.int32)
            loc_ff = tf.concat([loc_f, loc_f[:, -2:0:-1]], 1)
            kg_pred = tf.math.multiply(self.kg_raw, tf.tile(tf.reshape(loc_ff, [-1, self.model_size, 1, 1]), k))
            mse = tf.math.reduce_mean(tf.square(kg_true - kg_pred))
            return mse

    class XtLoss(keras.losses.Loss):
        def __init__(self, model_size, nobs, batch_size, name="xt_loss"):
            super().__init__(name=name)
            self.model_size = model_size
            self.nobs = nobs
            self.batch_size = batch_size

        def call(self, xt_xf_inc, kg_pred):
            xt = xt_xf_inc[:, 0:self.model_size, :]
            xf = xt_xf_inc[:, self.model_size:self.model_size * 2, :]
            inc = xt_xf_inc[:, self.model_size * 2:self.model_size * 2 + self.nobs, :]
            kg_pred = tf.squeeze(kg_pred)
            xa = xf + tf.matmul(kg_pred, inc)
            mse = tf.math.reduce_mean(tf.square(xt - xa))
            return mse

    class CustomMSE(keras.losses.Loss):
        def __init__(self, mean, std, name="custom_mse"):
            super().__init__(name=name)
            # self.y_pred_norm = y_pred_norm
            self.mean = mean
            self.std = std

        def call(self, y_true, y_pred):
            y_true_norm = tf.divide((y_true - self.mean), self.std)
            y_pred_norm = tf.divide((y_pred - self.mean), self.std)
            mse = tf.math.reduce_mean(tf.square(y_true_norm - y_pred_norm))
            return mse

    class PSNRLoss(keras.losses.Loss):
        def __init__(self, kg_raw, maxi=1.0, name="PSNR"):
            super().__init__(name=name)
            self.maxi = maxi

        def call(self, kg_true, kg_pred):
            rmse = tf.math.sqrt(tf.math.reduce_mean(tf.square(kg_true - kg_pred)))
            psnr = 20 * tf.math.log(self.maxi / rmse) / tf.math.log(tf.constant(10, dtype=rmse.dtype))

            return psnr

    class PSNR(tf.keras.metrics.Metric):
        def __init__(self, maxi=1.0, name='PSNR'):
            super().__init__(name=name)
            self.maxi = maxi
            self.psnr = self.add_weight(name='psnr', initializer='zeros')

        def update_state(self, kg_true, kg_pred, sample_weight=None):
            kg_true = tf.cast(kg_true, tf.float32)
            rmse = tf.math.sqrt(tf.math.reduce_mean(tf.square(kg_true - kg_pred)))
            self.psnr.assign(20 * tf.math.log(self.maxi / rmse) / tf.math.log(tf.constant(10, dtype=rmse.dtype)))

        def result(self):
            return self.psnr

        def reset_states(self):
            self.psnr.assign(0)

    gd_norm = GradientNorm()
    psnr_metric = PSNR()
    loss_tracker = keras.metrics.Mean(name="loss")

    class CustomModel(keras.Model):
        def train_step(self, data):
            # Unpack the data. Its structure depends on your model and
            # on what you pass to `fit()`.
            x, y = data

            with tf.GradientTape() as tape:
                y_pred = self(x, training=True)  # Forward pass
                # Compute the loss value
                # (the loss function is configured in `compile()`)
                loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
                # loss = keras.losses.mean_squared_error(y, y_pred)

            # Compute gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)

            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

            # Update metrics (includes the metric that tracks the loss)
            # self.compiled_metrics.reset_states()
            # self.compiled_metrics.update_state(y, gradients)
            self.compiled_metrics.update_state(y, y_pred)
            # gd_norm.update_state(y, gradients)
            # psnr_metric.update_state(y, y_pred)
            # loss_tracker.update_state(loss)

            # Return a dict mapping metric names to current value
            return_metrics = {m.name: m.result() for m in self.metrics}
            return return_metrics

        # @property
        # def metrics(self):
        #     # We list our `Metric` objects here so that `reset_states()` can be
        #     # called automatically at the start of each epoch
        #     # or at the start of `evaluate()`.
        #     # If you don't implement this property, you have to call
        #     # `reset_states()` yourself at the time of your choosing.
        #     return [loss_tracker, gd_norm, psnr_metric]

    class Normalization(layers.Layer):
        def __init__(self, mean, std):
            super(Normalization, self).__init__()
            self.mean = mean
            self.std = std

        def call(self, inputs):
            return tf.divide((inputs - self.mean), self.std)

    class DeNormalization(layers.Layer):
        def __init__(self, mean, std):
            super(DeNormalization, self).__init__()
            self.mean = mean
            self.std = std

        def call(self, inputs):
            return tf.multiply(inputs, self.std) + self.mean

    def get_dataset_partitions(ds, ds_size, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True,
                               shuffle_size=10000):
        assert (train_split + test_split + val_split) == 1

        if shuffle:
            # Specify seed to always have the same split distribution between runs
            ds = ds.shuffle(shuffle_size, seed=12)

        train_size = int(train_split * ds_size)
        val_size = int(val_split * ds_size)

        train_ds = ds.take(train_size)
        val_ds = ds.skip(train_size).take(val_size)
        test_ds = ds.skip(train_size).skip(val_size)

        return train_ds, val_ds, test_ds

    # -------------------- model setting --------------------
    model_size = 960
    nobs = 240
    obs_dens = 4
    loc_size = 481

    # create a model
    tf.keras.backend.set_floatx('float32')
    tf.keras.backend.floatx()
    inputs = keras.Input(shape=(model_size, nobs, 1), name="kg_raw")
    p1 = PeriodicPadding2D(padding=(7, 4))(inputs)
    x1 = layers.Conv2D(filters=128, kernel_size=(9, 3), strides=(1, 1), kernel_initializer='glorot_uniform',
                       activation='relu', padding='valid', use_bias=True)(p1)
    # p1 = layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid')(x1)
    x2 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='glorot_uniform',
                       activation='relu', padding='valid', use_bias=True)(x1)
    # p2 = layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid')(x2)
    x3 = layers.Conv2D(filters=1, kernel_size=(5, 5), strides=(1, 1), kernel_initializer='glorot_uniform',
                       activation='linear', padding='valid', use_bias=True)(x2)
    x4 = layers.Flatten()(x3)
    # x5 = layers.Dense(240, name='Dense1')(x4)
    x5 = layers.Dense(loc_size, name="loc_f")(x4)
    outputs = Loc_by_1D(obs_dens=obs_dens, model_size=model_size, nobs=nobs, batch_size=kwargs['batch_size'])([inputs, x5])
    model = CustomModel(inputs=inputs, outputs=outputs)

    # compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-6),
        loss=XtLoss(model_size, nobs, kwargs['batch_size']),
        # loss=tf.keras.losses.MeanSquaredError(),
        # experimental_run_tf_function=False,
        metrics=[GradientNorm()],
    )

    return model
