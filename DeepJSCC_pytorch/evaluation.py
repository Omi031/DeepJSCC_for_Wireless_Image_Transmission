import argparse
import datetime
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataloader import dataloader
from models import DeepJSCC
from torch.optim import lr_scheduler
from utils.metrics import CalculatePSNR, CalculateLPIPS
from tqdm.contrib import tenumerate
from tqdm import tqdm
from utils.utils import dB2linear
from utils.log import Logging
from utils.plot import show_and_save
from torchvision.utils import make_grid

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument("--ch", default="AWGN")
parser.add_argument("--c", default=4)
parser.add_argument("--weight")
parser.add_argument("--result_dir", default="")
parser.add_argument("--snr", default="")
parser.add_argument("--epochs", default="")
parser.add_argument("--note", default="")
args = parser.parse_args()


if args.ch == "SRF":
    ch = "SRF"
elif args.ch == "AWGN":
    ch = "AWGN"
else:
    raise Exception

SNR = str(args.snr)
epochs = str(args.epochs)
note = args.note
c = int(args.c)
result_dir = args.result_dir

datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


np.random.seed(42)

# results_dir = "results"

# result_dir = os.path.join(results_dir, f"{ch}_{datetime}")
# os.makedirs(result_dir, exist_ok=True)
os.makedirs(r".\%s\images" % result_dir, exist_ok=True)
os.makedirs(r".\%s\weight" % result_dir, exist_ok=True)

P = 1
PP = 8**2
n = 32 * 32 * 3
k = PP * c / 2
k_n = k / n


info_test = [
    "file: %s" % os.path.basename(__file__),
    "process: test",
    "date: %s" % datetime,
    "channel: %s" % ch,
    "SNR_train: %s" % SNR,
    "epoch: %d" % epochs,
    "c: %d" % c,
    "k/n: %f" % k_n,
    "note: %s" % note,
]
heads_test = ["SNR", "MSE", "PSNR", "LPIPS"]
logging_test = Logging(info=info_test, heads=heads_test, dir=result_dir, name="result")


# load data
subset = True
batch_size = 64
test_loader = dataloader(train=False, subset=subset)


deepjscc = DeepJSCC(ch, k, P, c).to(device)
criterion = nn.MSELoss()

# metrics
calculate_psnr = CalculatePSNR(val=[0, 1])
calculate_lpips = CalculateLPIPS(val=[0, 1])

test_vis = next(iter(test_loader))[0].to(device)

show_and_save(r".\%s\images\test.png" % result_dir, make_grid(test_vis.cpu(), 8))


format2 = "SNR: %d\tPSNR: %.2f\tLPIPS: %.4g"


deepjscc.eval()

for i in tqdm(range(0, 21, 2), desc="Test"):
    SNR = i
    N = dB2linear(SNR, P)
    test_batch_loss, test_batch_psnr, test_batch_lpips = [], [], []
    mse_list, psnr_list, lpips_list = [], [], []
    for i in range(10):
        with torch.no_grad():
            for i, (x, _) in tenumerate(test_loader):
                x = x.to(device)
                x_hat = deepjscc(x, N)
                mse = criterion(x_hat, x)
                psnr = calculate_psnr(x, x_hat)
                lpips = calculate_lpips(x, x_hat)
                test_batch_loss.append(mse.item())
                test_batch_psnr.append(psnr.item())
                test_batch_lpips.append(lpips.item())

        mse_list.append(np.mean(test_batch_loss))
        psnr_list.append(np.mean(test_batch_psnr))
        lpips_list.append(np.mean(test_batch_lpips))
    mse_test = np.mean(mse_list)
    psnr_test = np.mean(psnr_list)
    lpips_test = np.mean(lpips_list)
    contents = [SNR, psnr_test, lpips_test]
    logging_test(contents)
    tqdm.write(format2 % tuple(contents))
