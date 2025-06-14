import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

"""
Normalization
  Normalize pixel values according to power constraint P.

k       : Number of pixels per image
P       : Power constraint (default P = 1)
x_norm  : Norm of pixel values per image
p       : Poser per image
"""


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


# class Normalization(layers.Layer):
#   def __init__(self, k, P, **kwargs):
#     super(Normalization, self).__init__()
#     self.k = k
#     self.P = P
#   def call(self, x):
#     x_shape = tf.shape(x)
#     x_norm = tf.norm(x, axis=[1, 2, 3])
#     # x_norm = tf.expand_dims(x_norm, axis=1)
#     # x_norm = tf.expand_dims(x_norm, axis=2)
#     # x_norm = tf.expand_dims(x_norm, axis=3)
#     x_norm = tf.broadcast_to(x_norm, x_shape)
#     # x_norm = tf.tile(x_norm, [1,x_shape[1], x_shape[2], 1])

#     sqrt_kP = np.sqrt(self.k * self.P)
#     y = sqrt_kP*tf.divide(x, x_norm)
#     return y
#   def get_config(self):
#     config = super(Normalization, self).get_config()
#     config.update({
#       'k': self.k,
#       'P': self.P
#     })
#     return config
#   @classmethod
#   def from_config(cls, config):
#     return cls(**config)

"""
AWGN channel
x       : input
          shape = (Batch,8,8,c)

y       : output
          y = x + noise
          shape = (Batch,8,8,c)

noise   : AWGN noise
          noise~CN(0,N)
          N = P/(10^(SNR/10))

stddev  : Standard deviation of real noise distribution
N       : Variance of complex noise distribution
"""


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


# class Slow_Rayleigh_Fading_Channel(layers.Layer):
#   def __init__(self, **kwargs):
#     super(Slow_Rayleigh_Fading_Channel, self).__init__()
#     self.stddevs = np.sqrt(0.5)
#     idx_e = tf.constant([[[1,0]]], dtype=tf.float32)
#     idx_o = tf.constant([[[0,1]]], dtype=tf.float32)
#     self.idx_e = tf.expand_dims(idx_e, -1)
#     self.idx_o = tf.expand_dims(idx_o, -1)
#   def call(self, x):
#     with tf.name_scope('Slow_Rayleigh_Fading_Channel'):
#       x_shape = tf.shape(x)
#       zeros = tf.zeros(shape=[x_shape[0], x_shape[1], 1, x_shape[3]])
#       x_i = tf.concat([zeros, x], axis=2)
#       x_q = tf.concat([x, zeros], axis=2)
#       print(x_i.shape)

#       h_i = tf.random.normal(mean=0.0, stddev=self.stddevs, dtype=tf.float32)
#       h_q = tf.random.normal(mean=0.0, stddev=self.stddevs, dtype=tf.float32)


#     return x
#   @classmethod
#   def from_config(cls, config):
#     return cls(**config)

"""
Slow Rayleigh Fading Channel
x   : input 
      shape = (Batch,8,8,c)

y   : output
      shape = (Batch,8,8,c)
      y = h*x

h   : fading coefficient
      h~CN(0,1)

h_i : I components of h
      h_i~N(0,0.5)

h_q : Q components of h
      h_q~N(0,0.5)
"""


class Slow_Rayleigh_Fading_Channel(layers.Layer):
    def __init__(self, N, **kwargs):
        super(Slow_Rayleigh_Fading_Channel, self).__init__()
        # Slow Rayleigh fading
        # Fading variance is 1/2 because we consider real values (not complex)
        self.stddevs = np.sqrt(1)
        # Matrix for extracting even/odd columns of feature maps
        idx_e = tf.constant([[[1, 0]]], dtype=tf.float32)
        idx_o = tf.constant([[[0, 1]]], dtype=tf.float32)
        self.idx_e = tf.expand_dims(idx_e, -1)
        self.idx_o = tf.expand_dims(idx_o, -1)

        #

    def call(self, x):
        with tf.name_scope("Slow_Rayleigh_Fading_Channel"):
            x_shape = tf.shape(x)
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
        return y

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# class Slow_Rayleigh_Fading_Channel(layers.Layer):
#   def __init__(self, **kwargs):
#     super(Slow_Rayleigh_Fading_Channel, self).__init__()
#     self.stddevs = np.sqrt(0.5)
#   def call(self, x):
#     with tf.name_scope('Slow_Rayleigh_Fading_Channel'):
#       x_shape = tf.shape(x)
#       half = x_shape[3]//2
#       h_r = tf.random.normal([x_shape[0], 1, 1, 1], mean=0.0, stddev=self.stddevs, dtype=tf.float32)
#       h_i = tf.random.normal([x_shape[0], 1, 1, 1], mean=0.0, stddev=self.stddevs, dtype=tf.float32)
#       h_r = tf.tile(h_r, [1, x_shape[1], x_shape[2], half])
#       h_i = tf.tile(h_i, [1, x_shape[1], x_shape[2], half])
#       x_f = tf.math.multiply(h_r, x[:,:,:,:half]) - tf.math.multiply(h_i, x[:,:,:,half:])
#       x_s = tf.math.multiply(h_i, x[:,:,:,:half]) + tf.math.multiply(h_r, x[:,:,:,half:])
#       y = tf.concat([x_f, x_s], axis=-1)
#     return y
#   @classmethod
#   def from_config(cls, config):
#     return cls(**config)

# class Slow_Rayleigh_Fading_Channel(layers.Layer):
#   def __init__(self, **kwargs):
#     super(Slow_Rayleigh_Fading_Channel, self).__init__()
#   def random_normal(self):
#     return tf.random.normal([1, 1], mean=0.0, stddev=1.0, dtype=tf.float32)

#   @tf.function
#   def call(self, x):
#     with tf.name_scope('Slow_Rayleigh_Fading_Channel'):
#       x_shape = tf.shape(x)
#       x_shape = tf.cast(x_shape, dtype='int32')
#       x_real = x[:,:,:,0::2]
#       x_imag = x[:,:,:,1::2]
#       x_c = tf.complex(x_real, x_imag)
#       x_c = tf.keras.backend.eval(x_c)
#       y_c = []
#       # for i in range(tf.shape(x_c)[0]):
#       #   h = tf.complex(self.random_normal(), self.random_normal())
#       #   y_c.append(h*x_c[i])
#       for x_c_0 in x_c:
#         h = tf.complex(self.random_normal(), self.random_normal())
#         y_c.append(h*x_c_0)
#       y_c = tf.convert_to_tensor(y_c)
#       print(y_c.shape)
#       y = tf.TensorArray(dtype=tf.float32, size=int(x.shape[3]))

#       y_real = tf.math.real(y_c)
#       y_imag = tf.math.imag(y_c)
#       for i in range(y_c.shape[3]):
#         y = y.write(2*i, y_real[:,:,:,i])
#         y = y.write(2*i+1, y_imag[:,:,:,i])
#       y = y.stack()
#       y = tf.transpose(y, perm=[1,2,3,0])


#     return y
#   @classmethod
#   def from_config(cls, config):
#     return cls(**config)


# class Slow_Rayleigh_Fading_Channel(layers.Layer):
#   def __init__(self, **kwargs):
#     super(Slow_Rayleigh_Fading_Channel, self).__init__()
#   def call(self, x, **kwargs):
#     with tf.name_scope('Flat_Fasing_Channel'):
#       H = tf.random.normal(tf.shape(1), mean=0.0, stddev=1.0, dtype=tf.float32)
#       z = H*x
#     return z

# class Normalization(layers.Layer):
#   def __init__(self, k, P, **kwargs):
#     super(Normalization, self).__init__()
#     self.k = k
#     self.P = P
#   def call(self, x):
#     # float to complex
#     x_real = x[:,:,:,0::2]
#     x_imag = x[:,:,:,1::2]
#     x_c = tf.complex(x_real, x_imag)

#     x_ct = tf.math.conj(x_c)
#     kP = tf.cast(self.k * self.P, dtype=tf.complex64)
#     # tf.print(self.k * self.P)
#     # denominator = tf.sqrt(tf.maximum(tf.reduce_sum(x_ct * x_c, axis=-1, keepdims=True), 1e-6))
#     # norm_factor = tf.sqrt(tf.maximum(kP, 1e-6))
#     # tf.print(x)
#     y_c = tf.multiply(kP, tf.divide(x_c, tf.multiply(x_ct, x_c)))
#     x_shape = x.shape

#     y = tf.TensorArray(dtype=tf.float32, size=int(x.shape[3]))

#     y_real = tf.math.real(y_c)
#     y_imag = tf.math.imag(y_c)
#     for i in range(y_c.shape[3]):
#       y = y.write(2*i, y_real[:,:,:,i])
#       y = y.write(2*i+1, y_imag[:,:,:,i])
#     y = y.stack()
#     y = tf.transpose(y, perm=[1,2,3,0])
#     # tf.print(tf.shape(y))
#     print(y.shape)

#     return y
#   def get_config(self):
#     config = super(Normalization, self).get_config()
#     config.update({
#       'k': self.k,
#       'P': self.P
#     })
#     return config

# class Normalization(layers.Layer):
# def __init__(self, k, P, **kwargs):
#   super(Normalization, self).__init__()
#   self.k = tf.cast(k, dtype=tf.complex128)
#   self.P = tf.cast(P, dtype=tf.complex128)
# def call(self, z_tilta):
#   z_tilta = tf.cast(z_tilta, dtype=tf.complex128)
#   z_conjugateT = tf.math.conj(z_tilta, name='z_ConjugateTrans')
#   #Square root of k and P
#   sqrt1 = tf.dtypes.cast(tf.math.sqrt(self.k*self.P, name='NormSqrt1'), dtype='complex128',name='ComplexCastingNorm')
#   sqrt2 = tf.math.sqrt(z_conjugateT*z_tilta, name='NormSqrt2')#Square root of z_tilta* and z_tilta.

#   div = tf.math.divide(z_tilta,sqrt2, name='NormDivision')
#   #calculating channel input
#   z = tf.math.multiply(sqrt1,div, name='Z')
#   z = tf.cast(z, dtype='float32')
#   return z
# def get_config(self):
#   config = super(Normalization, self).get_config()
#   config.update({
#     'k': self.k,
#     'P': self.P
#   })
#   return config


# class Normalization(layers.Layer):
#   def __init__(self, k, P, **kwargs):
#     super(Normalization, self).__init__()
#     self.k = k
#     self.P = P
#   def call(self, x):
#     x = tf.cast(x, dtype=tf.float32)
#     x_ct = tf.transpose(x, perm=[0, 2, 1, 3], conjugate=True)
#     kP = tf.cast(self.k * self.P, dtype='float32')
#     denominator = tf.sqrt(tf.maximum(tf.reduce_sum(x_ct * x, axis=-1, keepdims=True), 1e-6))
#     z = tf.multiply(tf.sqrt(kP), tf.divide(x, denominator))
#     return z
#   def get_config(self):
#     config = super(Normalization, self).get_config()
#     config.update({
#       'k': self.k,
#       'P': self.P
#     })
#     return config


# class Modulation(layers.Layer):
#   def __init__(self, **kwargs):
#     super(Modulation, self).__init__()
#   def call(self, x):
#     # x_shape = tf.shape(x)
#     # y = np.zeros((int(x_shape[0]), int(x_shape[1]), int(x_shape[2]), int(x.shape[3]/2)))
#     # y = tf.TensorArray(dtype=tf.complex64, size=int(x.shape[3]/2))
#     # for i in range(int(x.shape[3]/2)):
#     #   y_real = x[:,:,:,2*i]
#     #   y_imag = x[:,:,:,2*i+1]
#     #   y.write(i, tf.complex(y_real, y_imag))
#     # y = y.stack()
#     # print(tf.shape(y))
#     # print(y.shape)
#     # y = tf.transpose(y, perm=[1,2,3,0])
#     # tf.print(tf.shape(y))
#     # print(y.shape)

#     y_real = tf.cast(x[:,:,:,0::2], tf.complex128)
#     y_imag = tf.cast(x[:,:,:,1::2], tf.complex128)

#     # y = tf.complex(y_real, y_imag)
#     y = y_real + tf.multiply(1j, y_imag)

#     return y

# class Demodulation(layers.Layer):
#   def __init__(self, **kwargs):
#     super(Demodulation, self).__init__()
#   def call(self, x):
#     y = tf.TensorArray(dtype=tf.float32, size=int(x.shape[3]*2))
#     y_real = x.real
#     y_imag = x.imag
#     for i in range(int(x.shape[3]*2)):
#       y.write(2*i, y_real[:,:,:,i])
#       y.write(2*i+1, y_imag[:,:,:,i])
#     y = y.stack()
#     y = tf.transpose(y, perm=[1,2,3,0])
#     tf.print(tf.shape(y))
#     print(y.shape)
#     return y
