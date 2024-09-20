import lpips
import tensorflow as tf
from lpips_tensorflow import lpips_tf


# PSNR
class CalculatePSNR:
    def __init__(self, val=[0, 1]):
        self.max_val = abs(val[1] - val[0])

    def __call__(self, x, x_hat):
        return tf.image.psnr(x, x_hat, max_val=self.max_val)


# LPIPS
class CalculateLPIPS:
    def __init__(self, val=[-1, 1], net="vgg"):
        self.net = net
        self.val = val
        self.val_new = [0, 1]

    def __call__(self, x, x_hat):

        if self.val != self.val_new:
            x = linear_conversion(x, self.val, self.val_new)
            x_hat = linear_conversion(x_hat, self.val, self.val_new)

        return lpips_tf.lpips(x, x_hat, net=self.net)


def linear_conversion(x, val_pre, val_new):
    min_pre, max_pre = val_pre
    min_new, max_new = val_new
    return (x - min_pre) * (max_new - min_new) / (max_pre - min_pre) + min_new


# def tf_tensor2pt_tensor(x):
#     x = tf.transpose(x, perm=[0, 3, 1, 2])
#     x = torch.Tensor(x.numpy())
#     return x
