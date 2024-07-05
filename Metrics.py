import tensorflow as tf

# PSNR
def PSNR(y_true, y_pred):
  return tf.image.psnr(y_true, y_pred, max_val=1.0)