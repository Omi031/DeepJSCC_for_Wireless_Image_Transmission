import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, callbacks, datasets, layers, models
from dataclasses import dataclass


class LossFunction:
    def __call__(self, mean, logvar, z, x_hat, x):
        rc_loss = tf.reduce_mean(
            tf.reduce_sum(keras.losses.binary_crossentropy(x, x_hat), axis=(1, 2))
        )
        kl_loss = tf.reduce_mean(
            tf.reduce_sum(
                -0.5 * (1 + logvar - tf.square(mean) - tf.exp(logvar)), axis=1
            )
        )
        loss = rc_loss + kl_loss
        return rc_loss, kl_loss, loss


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
        self.mean_dense = layers.Dense(z_dim)
        self.logvar_dense = layers.Dense(z_dim)

    def call(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.flatten(x)
        x = self.dense1(x)
        mean = self.mean_dense(x)
        logvar = self.logvar_dense(x)
        return mean, logvar


class Sampling(layers.Layer):
    def __init__(self):
        super(Sampling, self).__init__()

    def call(self, mean, logvar):
        mean_shape = tf.shape(mean)
        epsilon = tf.random.normal(shape=mean_shape)
        z = mean + tf.exp(0.5 * logvar) * epsilon
        return z


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


class Decoder(layers.Layer):
    def __init__(self, z_dim):
        super(Decoder, self).__init__()
        self.dense1 = layers.Dense(8 * 8 * 128)
        self.relu = layers.LeakyReLU()
        self.reshape = layers.Reshape(target_shape=(8, 8, 128))
        self.deconv1 = layers.Conv2DTranspose(64, (5, 5), strides=1, padding="same")
        self.deconv2 = layers.Conv2DTranspose(32, (5, 5), strides=1, padding="same")
        self.deconv3 = layers.Conv2DTranspose(16, (5, 5), strides=2, padding="same")
        self.deconv4 = layers.Conv2DTranspose(
            3, (5, 5), strides=2, padding="same", activation="sigmoid"
        )

    def call(self, x):
        x = self.relu(self.dense1(x))
        x = self.reshape(x)
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.relu(self.deconv3(x))
        x = self.deconv4(x)
        return x


class VAE(Model):
    def __init__(self, z_dim, N, optimizer, ch="AWGN"):
        super(VAE, self).__init__()
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)
        self.sampling = Sampling()
        if ch == "AWGN":
            self.channel = AWGN_Channel(N)
        self.loss_function = LossFunction()
        self.optimizer = optimizer

    def call(self, x, train=True):
        mean, logvar = self.encoder(x)
        if train == True:
            z = self.sampling(mean, logvar)

        else:
            z = mean
        x_hat = self.decoder(z)
        return mean, logvar, z, x_hat

    @tf.function
    def train_step_function(self, x):
        with tf.GradientTape() as tape:
            mean, logvar, z, x_hat = self.call(x)
            rc_loss, kl_loss, loss = self.loss_function(mean, logvar, z, x_hat, x)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return rc_loss, kl_loss, loss

    def train_step(self, x):
        rc_loss, kl_loss, loss = self.train_step_function(x)
        return rc_loss, kl_loss, loss

    def test_step(self, x):
        mean, logvar, z, x_hat = self.call(x, train=False)
        rc_loss, kl_loss, loss = self.loss_function(mean, logvar, z, x_hat, x)
        return x_hat, rc_loss, kl_loss, loss

    def predict(self, x):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        x = tf.expand_dims(x, axis=0)
        mean, logvar, z, x_hat = self.call(x)
        z = tf.squeeze(z, [0])
        x_hat = tf.squeeze(x_hat, [0])
        return z, x_hat


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
