import lpips
import torch
from torchmetrics.image import PeakSignalNoiseRatio


# PSNR
class CalculatePSNR:
    def __init__(self, val_pre=[-1, 1], val_new=[0, 1]):
        self.val_pre = val_pre
        self.val_new = val_new
        data_range = val_new[1] - val_new[0]
        self.psnr_fn = PeakSignalNoiseRatio(data_range=data_range)

    def __call__(self, true, pred):
        if self.val_pre != self.val_new:
            true = linear_conversion(true, self.val_pre, self.val_new)
            pred = linear_conversion(pred, self.val_pre, self.val_new)

        return self.psnr_fn(true, pred)


# LPIPS
class CalculateLPIPS:
    def __init__(self, net="vgg", val_pre=[-1, 1], val_new=[-1, 1]):
        self.lpips_fn = lpips.LPIPS(net=net)
        self.val_pre = val_pre
        self.val_new = val_new

    def __call__(self, true, pred):
        if self.val_pre != self.val_new:
            true = linear_conversion(true, self.val_pre, self.val_new)
            pred = linear_conversion(pred, self.val_pre, self.val_new)

        return torch.mean(self.lpips_fn(true, pred))


def linear_conversion(x, val_pre, val_new):
    min_pre, max_pre = val_pre
    min_new, max_new = val_new
    return (x - min_pre) * (max_new - min_new) / (max_pre - min_pre) + min_new


# def tf_tensor2pt_tensor(x):
#     x = tf.transpose(x, perm=[0, 3, 1, 2])
#     x = torch.Tensor(x.numpy())
#     return x
