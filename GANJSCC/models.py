import tensorflow as tf
from tensorflow.keras import datasets, layers, models, callbacks, optimizers
import numpy as np


class Normalization(layers.Layer):
    def __init__(self, k, P, **kwargs):
        super(Normalization, self).__init__()
        self.k = k
        self.P = P

    def call(self, x):
        x_shape = tf.shape(x)
        x_norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=[1, 2, 3], keepdims=True))
        x_norm = tf.broadcast_to(x_norm, x_shape)
        p = np.sqrt(self.k * self.P)
        y = p * x / x_norm
        return y

    def get_config(self):
        config = super(Normalization, self).get_config()
        config.update({"k": self.k, "P": self.P})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class AWGN_Channel(layers.Layer):
    def __init__(self, N, **kwargs):
        super(AWGN_Channel, self).__init__()
        # Divde N by 2 because we consider real noise
        self.stddev = np.sqrt(N / 2)

    def call(self, x):
        with tf.name_scope("AWGN_Channel"):
            # AWGN noise
            noise = tf.random.normal(
                tf.shape(x), mean=0.0, stddev=self.stddev, dtype=tf.float32
            )
            y = x + noise
        return y

    def get_config(self):
        config = super(AWGN_Channel, self).get_config()
        config.update({"N": self.stddev})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Slow_Rayleigh_Fading_Channel(layers.Layer):
    def __init__(self, N, **kwargs):
        super(Slow_Rayleigh_Fading_Channel, self).__init__()
        # =====================
        # Slow Rayleigh fading
        # =====================
        # Fading variance is 1/2 because we consider real values (not complex)
        self.stddevs = np.sqrt(0.5)
        # Matrix for extracting even/odd columns of feature maps
        idx_e = tf.constant([[[1, 0]]], dtype=tf.float32)
        idx_o = tf.constant([[[0, 1]]], dtype=tf.float32)
        self.idx_e = tf.expand_dims(idx_e, -1)
        self.idx_o = tf.expand_dims(idx_o, -1)

        # =====
        # AWGN
        # =====
        self.stddev = np.sqrt(N / 2)

    def call(self, x):
        with tf.name_scope("Slow_Rayleigh_Fading_Channel"):
            x_shape = tf.shape(x)

            # =====================
            # Slow Rayleigh fading
            # =====================

            # Matrix for extracting even/odd columns of feature maps
            idx_e = tf.tile(
                self.idx_e, [x_shape[0], x_shape[1], x_shape[2] // 2, x_shape[3]]
            )
            idx_o = tf.tile(
                self.idx_o, [x_shape[0], x_shape[1], x_shape[2] // 2, x_shape[3]]
            )
            # Define even/odd columns of feature maps of x as I/Q components
            x_i = x[:, :, 0::2, :]
            x_q = x[:, :, 1::2, :]
            # Rayleigh fading coefficient
            h_i = tf.random.normal(
                [x_shape[0], 1, 1, 1], mean=0.0, stddev=self.stddevs, dtype=tf.float32
            )
            h_q = tf.random.normal(
                [x_shape[0], 1, 1, 1], mean=0.0, stddev=self.stddevs, dtype=tf.float32
            )
            # y = h*x
            y_i = h_i * x_i - h_q * x_q
            y_q = h_q * x_i + h_i * x_q
            # Combine I and Q components
            y_i2 = tf.gather(y_i, [0, 0, 1, 1, 2, 2, 3, 3], axis=2)
            y_q2 = tf.gather(y_q, [0, 0, 1, 1, 2, 2, 3, 3], axis=2)
            y = y_i2 * idx_e + y_q2 * idx_o

            # =====
            # AWGN
            # =====

            y_norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=[1, 2, 3], keepdims=True))

            # AWGN noise
            noise = (
                tf.random.normal(
                    tf.shape(x), mean=0.0, stddev=self.stddev, dtype=tf.float32
                )
                * y_norm
            )

            z = y + noise
        return z

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def deepjscc(c, k, P, N, slow_rayleigh_fading=False):
    if slow_rayleigh_fading == True:
        channel = Slow_Rayleigh_Fading_Channel(N)
    else:
        channel = AWGN_Channel(N)
    model = models.Sequential(name="DeepJSCC")
    # encorder
    model.add(
        layers.Conv2D(16, (5, 5), strides=2, padding="same", input_shape=(32, 32, 3))
    )
    model.add(layers.PReLU())

    model.add(layers.Conv2D(32, (5, 5), strides=2, padding="same"))
    model.add(layers.PReLU())

    model.add(layers.Conv2D(32, (5, 5), strides=1, padding="same"))
    model.add(layers.PReLU())

    model.add(layers.Conv2D(32, (5, 5), strides=1, padding="same"))
    model.add(layers.PReLU())

    model.add(layers.Conv2D(c, (5, 5), strides=1, padding="same"))
    model.add(layers.PReLU())

    model.add(Normalization(k, P))

    # add channel noise
    model.add(channel())

    # encorder
    model.add(layers.Conv2DTranspose(32, (5, 5), strides=1, padding="same"))
    model.add(layers.PReLU())
    model.add(layers.Conv2DTranspose(32, (5, 5), strides=1, padding="same"))
    model.add(layers.PReLU())
    model.add(layers.Conv2DTranspose(32, (5, 5), strides=1, padding="same"))
    model.add(layers.PReLU())
    model.add(layers.Conv2DTranspose(16, (5, 5), strides=2, padding="same"))
    model.add(layers.PReLU())
    model.add(
        layers.Conv2DTranspose(
            3,
            (5, 5),
            strides=2,
            padding="same",
            activation="sigmoid",
        )
    )
    return model


class DeepJSCC:
    def __init__(self, c, k, P, N, slow_rayleigh_fading=False):
        super(DeepJSCC, self).__init__()
        self.c = c
        self.k = k
        self.P = P
        self.N = N
        if slow_rayleigh_fading == True:
            self.channel = Slow_Rayleigh_Fading_Channel(N)
        else:
            self.channel = AWGN_Channel(N)

    def model(self):
        model = models.Sequential(name="DeepJSCC")
        # encorder
        model.add(
            layers.Conv2D(
                16, (5, 5), strides=2, padding="same", input_shape=(32, 32, 3)
            )
        )
        model.add(layers.PReLU())

        model.add(layers.Conv2D(32, (5, 5), strides=2, padding="same"))
        model.add(layers.PReLU())

        model.add(layers.Conv2D(32, (5, 5), strides=1, padding="same"))
        model.add(layers.PReLU())

        model.add(layers.Conv2D(32, (5, 5), strides=1, padding="same"))
        model.add(layers.PReLU())

        model.add(layers.Conv2D(self.c, (5, 5), strides=1, padding="same"))
        model.add(layers.PReLU())

        model.add(Normalization(self.k, self.P))

        # add channel noise
        model.add(self.channel())

        # encorder
        model.add(layers.Conv2DTranspose(32, (5, 5), strides=1, padding="same"))
        model.add(layers.PReLU())
        model.add(layers.Conv2DTranspose(32, (5, 5), strides=1, padding="same"))
        model.add(layers.PReLU())
        model.add(layers.Conv2DTranspose(32, (5, 5), strides=1, padding="same"))
        model.add(layers.PReLU())
        model.add(layers.Conv2DTranspose(16, (5, 5), strides=2, padding="same"))
        model.add(layers.PReLU())
        model.add(
            layers.Conv2DTranspose(
                3,
                (5, 5),
                strides=2,
                padding="same",
                activation="sigmoid",
            )
        )
        return model


class Discriminator:
    def __init__(self):
        super(Discriminator, self).__init__()

    def __call__(self):
        model = models.Sequential(name="Discriminator")

        model.add(layers.Conv2D(16, (5, 5), strides=1, padding="same"))
        model.add(layers.PReLU())
        model.add(layers.Conv2D(32, (5, 5), strides=2, padding="same"))
        model.add(layers.PReLU())
        model.add(layers.Conv2D(64, (5, 5), strides=2, padding="same"))
        model.add(layers.PReLU())
        model.add(layers.Conv2D(128, (5, 5), strides=2, padding="same"))
        model.add(layers.PReLU())
        model.add(layers.Dense(512))
        model.add(layers.PReLU())
        model.add(layers.Dense(1))
        model.add(layers.Activation("sigmoid"))
        return model
