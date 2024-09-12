import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, callbacks, datasets, layers, models
from dataclasses import dataclass


class Normalization(layers.Layer):
    def __init__(self, k, P, **kwargs):
        super(Normalization, self).__init__()
        self.k = k
        self.P = P

    def call(self, x):
        x_shape = tf.shape(x)
        x_norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=[1], keepdims=True))
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


class Encoder(layers.Layer):
    def __init__(self, z_dim):
        super(Encoder, self).__init__()
        self.conv1 = layers.Conv2D(16, (5, 5), strides=2, padding="same")
        self.relu = layers.LeakyReLU()
        self.conv2 = layers.Conv2D(32, (5, 5), strides=2, padding="same")
        self.conv3 = layers.Conv2D(64, (5, 5), strides=1, padding="same")
        self.conv4 = layers.Conv2D(128, (5, 5), strides=1, padding="same")
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(4096)
        self.dense2 = layers.Dense(z_dim)

    def call(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.flatten(x)
        x = self.relu(self.dense1(x))
        x = self.dense2(x)

        return x


class Decoder(Model):
    def __init__(self, z_dim):
        super(Decoder, self).__init__()
        self.dense1 = layers.Dense(8 * 8 * 128)
        self.relu = layers.LeakyReLU()
        self.reshape = layers.Reshape(target_shape=(8, 8, 128))
        self.deconv1 = layers.Conv2DTranspose(64, (5, 5), strides=1, padding="same")
        self.deconv2 = layers.Conv2DTranspose(32, (5, 5), strides=1, padding="same")
        self.deconv3 = layers.Conv2DTranspose(16, (5, 5), strides=2, padding="same")
        self.deconv4 = layers.Conv2DTranspose(3, (5, 5), strides=2, padding="same")
        self.tanh = layers.Activation("tanh")

    def call(self, x):
        x = self.relu(self.dense1(x))
        x = self.reshape(x)
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.relu(self.deconv3(x))
        x = self.deconv4(x)
        x = self.tanh(x)

        return x


class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = layers.Conv2D(16, (5, 5), strides=1, padding="same")
        self.relu = layers.LeakyReLU()
        self.conv2 = layers.Conv2D(32, (5, 5), strides=2, padding="same")
        self.conv3 = layers.Conv2D(64, (5, 5), strides=2, padding="same")
        self.conv4 = layers.Conv2D(128, (5, 5), strides=2, padding="same")
        self.dense1 = layers.Dense(512)
        self.dense2 = layers.Dense(1)
        self.sigmoid = layers.Activation("sigmoid")

    def call(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.dense1(x))
        x = self.dense2(x)
        # x = self.sigmoid(x)
        return x


class DeepJSCC(Model):
    def __init__(self, z_dim, k, P, N, ch="AWGN"):
        super(DeepJSCC, self).__init__()
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)
        self.normalization = Normalization(k, P)
        if ch == "AWGN":
            self.channel = AWGN_Channel(N)

    def call(self, x):
        x = self.encoder(x)
        x = self.normalization(x)
        x = self.channel(x)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    z_dim = 512
    decoder = Decoder(z_dim)
    decoder.build(input_shape=(None, z_dim))
    decoder.call(tf.keras.Input(shape=(z_dim)))
    decoder.summary()

    deepjscc = DeepJSCC(z_dim, z_dim / 2, 1, 1)
    deepjscc.build(input_shape=(None, 32, 32, 3))
    deepjscc.call(tf.keras.Input(shape=(32, 32, 3)))
    deepjscc.summary()


# class VAE(tf.keras.Model):
#     def __init__(self, z_dim):
#         super(VAE, self).__init__()
#         self.encoder = tf.keras.Sequential(
#             [
#                 layers.InputLayer(input_shape=(32, 32, 3)),
#                 layers.Conv2D(16, (5, 5), strides=2, padding="same"),
#                 layers.PReLU(),
#                 layers.Conv2D(64, (5, 5), strides=1, padding="same"),
#                 layers.PReLU(),
#                 layers.Conv2D(128, (5, 5), strides=1, padding="same"),
#                 layers.PReLU(),
#                 layers.Flatten(),
#                 layers.Dense(4096),
#                 layers.PReLU(),
#                 layers.Dense(z_dim),
#                 layers.PReLU(),
#             ]
#         )
#         self.decoder = tf.keras.Sequential(
#             [
#                 layers.InputLayer(input_shape=(z_dim)),
#                 layers.Dense(8 * 8 * 128),
#                 layers.PReLU(),
#                 layers.Reshape(target_shape=(8, 8, 128)),
#                 layers.Conv2DTranspose(64, (5, 5), strides=1, padding="same"),
#                 layers.PReLU(),
#                 layers.Conv2DTranspose(32, (5, 5), strides=1, padding="same"),
#                 layers.PReLU(),
#                 layers.Conv2DTranspose(16, (5, 5), strides=1, padding="same"),
#                 layers.PReLU(),
#                 layers.Conv2DTranspose(
#                     3, (5, 5), strides=1, padding="same", activation="sigmoid"
#                 ),
#             ]
#         )
