import tensorflow as tf
import lpips
import torch


# PSNR
class PSNR:

    def __call__(self, y_true, y_pred):
        return tf.image.psnr(y_true, y_pred, max_val=1.0)


# LPIPS
class LPIPS:
    def __init__(self, net="vgg", val=[-1, 1]):
        self.lpips_fn = lpips.LPIPS(net=net)
        self.min = val[0]
        self.max = val[1]

    def __call__(self, true, pred):

        true = self.linear_conversion(true)
        pred = self.linear_conversion(pred)

        lpips = self.lpips_fn(true, pred)
        return lpips

    def linear_conversion(self, x):
        return 2 * (x - self.min) / (self.max - self.min) - 1


def tf_tensor2pt_tensor(x):
    x = tf.transpose(x, perm=[0, 3, 1, 2])
    x = torch.Tensor(x.numpy())
    return x
