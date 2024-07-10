import numpy as np
import tensorflow as tf
from tensorflow.keras import layers



class Normalization(layers.Layer):
  def __init__(self, k, P, **kwargs):
    super(Normalization, self).__init__()
    self.k = k
    self.P = P
  def call(self, x):
    x_shape = tf.shape(x)
    A_x = tf.reduce_sum(x, axis=[1,2,3], keepdims=True)
    A_x = tf.broadcast_to(A_x, x_shape)
    A_all = np.sqrt(self.k * self.P)
    y = A_all*x/A_x
    return y
  def get_config(self):
    config = super(Normalization, self).get_config()
    config.update({
      'k': self.k,
      'P': self.P
    })
    return config
  @classmethod
  def from_config(cls, config):
    return cls(**config)



class AWGN_Channel(layers.Layer):
  def __init__(self, N, **kwargs):
    super(AWGN_Channel, self).__init__()
    self.stddev = np.sqrt(N)
  def call(self, x):
    with tf.name_scope('AWGN_Channel'):
      noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=self.stddev, dtype=tf.float32)
      y = x + noise
    return y
  def get_config(self):
        config = super(AWGN_Channel, self).get_config()
        config.update({
          'N': self.stddev
        })
        return config
  @classmethod
  def from_config(cls, config):
    return cls(**config)



class Slow_Rayleigh_Fading_Channel(layers.Layer):
  def __init__(self, **kwargs):
    super(Slow_Rayleigh_Fading_Channel, self).__init__()
  def random_normal(self):
    return tf.random.normal([1, 1], mean=0.0, stddev=1.0, dtype=tf.float32)
  
  @tf.function
  def call(self, x):
    with tf.name_scope('Slow_Rayleigh_Fading_Channel'):
      x_shape = tf.shape(x)
      x_shape = tf.cast(x_shape, dtype='int32')
      x_real = x[:,:,:,0::2]
      x_imag = x[:,:,:,1::2]
      x_c = tf.complex(x_real, x_imag)
      x_c = tf.keras.backend.eval(x_c)
      y_c = []
      # for i in range(tf.shape(x_c)[0]):
      #   h = tf.complex(self.random_normal(), self.random_normal())
      #   y_c.append(h*x_c[i])
      for x_c_0 in x_c:
        h = tf.complex(self.random_normal(), self.random_normal())
        y_c.append(h*x_c_0)
      y_c = tf.convert_to_tensor(y_c)
      print(y_c.shape)
      y = tf.TensorArray(dtype=tf.float32, size=int(x.shape[3]))

      y_real = tf.math.real(y_c)
      y_imag = tf.math.imag(y_c)
      for i in range(y_c.shape[3]):
        y = y.write(2*i, y_real[:,:,:,i])
        y = y.write(2*i+1, y_imag[:,:,:,i])
      y = y.stack()
      y = tf.transpose(y, perm=[1,2,3,0])


    return y
  @classmethod
  def from_config(cls, config):
    return cls(**config)


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