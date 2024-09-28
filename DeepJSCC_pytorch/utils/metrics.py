import lpips
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# PSNR
class CalculatePSNR:
    def __init__(self, val=[-1, 1]):
        self.max = abs(val[1] - val[0])

    def __call__(self, x, x_hat):
        mse = torch.mean(torch.square(x - x_hat), dim=(1, 2, 3)).to(device)
        psnr = 10 * torch.log10(self.max**2 / mse).to(device)

        return psnr


# LPIPS
class CalculateLPIPS:
    def __init__(self, net="vgg", val=[-1, 1]):
        self.val = val
        self.lpips_fn = lpips.LPIPS(net=net).to(device)
        self.val_new = [-1, 1]

    def __call__(self, x, x_hat):
        if self.val != self.val_new:
            x = linear_conversion(x, self.val, self.val_new)
            x_hat = linear_conversion(x_hat, self.val, self.val_new)
        lpips = torch.squeeze(self.lpips_fn(x, x_hat))

        return lpips


def linear_conversion(x, val_pre, val_new):
    min_pre, max_pre = val_pre
    min_new, max_new = val_new
    return (x - min_pre) * (max_new - min_new) / (max_pre - min_pre) + min_new


if __name__ == "__main__":
    size = (64, 3, 32, 32)
    x = torch.rand(size=size)
    x_hat = torch.rand(size=size)

    calculate_psnr = CalculatePSNR(val=[0, 1])
    calculate_lpips = CalculateLPIPS(val=[0, 1])

    psnr = calculate_psnr(x, x_hat)
    lpips = calculate_lpips(x, x_hat)
