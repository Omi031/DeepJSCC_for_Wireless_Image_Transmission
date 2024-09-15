import tensorflow as tf
import lpips
import torch
import numpy as np


# PSNR
def PSNR(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)


class LPIPS:
    def __init__(self, net="vgg", val_pre=[-1, 1], val_new=[-1, 1]):
        self.lpips_fn = lpips.LPIPS(net=net)
        self.val_pre = val_pre
        self.val_new = val_new

    def __call__(self, true, pred):
        if self.val_pre != self.val_new:
            true = tf_tensor2pt_tensor(true)
            pred = tf_tensor2pt_tensor(pred)
            true = linear_conversion(true, self.val_pre, self.val_new)
            pred = linear_conversion(pred, self.val_pre, self.val_new)

        lpips = self.lpips_fn(true, pred)
        return lpips


def linear_conversion(x, val_pre, val_new):
    min_pre, max_pre = val_pre
    min_new, max_new = val_new
    return (x - min_pre) * (max_new - min_new) / (max_pre - min_pre) + min_new


@tf.function
def tf_tensor2pt_tensor(x):
    tf.config.experimental_run_functions_eagerly(True)
    x = tf.transpose(x, perm=[0, 3, 1, 2])
    # x = tf.convert_to_tensor(x)
    x = x.numpy()
    x = torch.Tensor(x)
    return x
