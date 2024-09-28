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
parser.add_argument("--snr", default=0)
parser.add_argument("--c", default=4)
parser.add_argument("--result_dir", default="")
args = parser.parse_args()


if args.ch == "SRF":
    ch = "SRF"
elif args.ch == "AWGN":
    ch = "AWGN"
else:
    raise Exception

SNR = int(args.snr)
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
N = dB2linear(SNR, P)
PP = 8**2
n = 32 * 32 * 3
k = PP * c / 2
k_n = k / n

info_train = [
    "file: %s" % os.path.basename(__file__),
    "process: train",
    "date: %s" % datetime,
    "channel: %s" % ch,
    "SNR_train: %ddB" % SNR,
    "c: %d" % c,
    "k/n: %f" % k_n,
    "note: left train, right, test",
]
heads_train = ["epoch", "MSE", "PSNR", "LPIPS", "MSE", "PSNR", "LPIPS"]
logging_train = Logging(info=info_train, heads=heads_train, dir=result_dir, name="log")

info_test = [
    "file: %s" % os.path.basename(__file__),
    "process: test",
    "date: %s" % datetime,
    "channel: %s" % ch,
    "SNR_train: %ddB" % SNR,
    "c: %d" % c,
    "k/n: %f" % k_n,
    "note: ",
]
heads_test = ["SNR", "MSE", "PSNR", "LPIPS"]
logging_test = Logging(info=info_test, heads=heads_test, dir=result_dir, name="result")


# load data
subset = True
train_loader = dataloader(train=True, subset=subset)
test_loader = dataloader(train=False, subset=subset)


epochs = 100
batch_size = 64
lr_1 = 1e-3
lr_2 = 1e-4
gamma = lr_2 / lr_1
iteration = 500000
lr_epoch = iteration * batch_size // len(train_loader)

deepjscc = DeepJSCC(ch, k, P, c).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(deepjscc.parameters(), lr=lr_1)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[lr_epoch], gamma=gamma)

# metrics
calculate_psnr = CalculatePSNR(val=[0, 1])
calculate_lpips = CalculateLPIPS(val=[0, 1])

test_vis = next(iter(test_loader))[0].to(device)

show_and_save(r".\%s\images\test.png" % result_dir, make_grid(test_vis.cpu(), 8))


mse_list = []
psnr_list = []
lpips_list = []
format1 = "%d\tMSE: %.4g\tPSNR: %.2f\tLPIPS: %.4g\tMSE: %.4g\tPSNR: %.2f\tLPIPS: %.4g"
format2 = "SNR: %d\tPSNR: %.2f\tLPIPS: %.4g"
for epoch in tqdm(range(epochs), desc="Train"):
    # train
    deepjscc.train()
    train_batch_loss, train_batch_psnr, train_batch_lpips = [], [], []
    for i, (x, _) in tenumerate(train_loader, leave=False, desc="train"):
        x = x.to(device)
        optimizer.zero_grad()
        x_hat = deepjscc(x, N)
        mse = criterion(x_hat, x)
        psnr = calculate_psnr(x, x_hat)
        lpips = calculate_lpips(x, x_hat)
        train_batch_loss.append(mse.item())
        train_batch_psnr.append(psnr.item())
        train_batch_lpips.append(lpips.item())
        mse.backward()
        optimizer.step()
        train_batch_loss.append(mse.item())

    mse_train = np.mean(train_batch_loss)
    psnr_train = np.mean(train_batch_psnr)
    lpips_train = np.mean(train_batch_lpips)

    scheduler.step()

    test_batch_loss = []
    test_batch_psnr = []
    test_batch_lpips = []

    # test
    deepjscc.eval()
    with torch.no_grad():
        for i, (x, _) in tenumerate(test_loader, leave=False, desc="test"):
            x = x.to(device)
            x_hat = deepjscc(x, N)
            mse = criterion(x_hat, x)
            psnr = calculate_psnr(x, x_hat)
            lpips = calculate_lpips(x, x_hat)
            test_batch_loss.append(mse.item())
            test_batch_psnr.append(psnr.item())
            test_batch_lpips.append(lpips.item())

    mse_test = np.mean(test_batch_loss)
    psnr_test = np.mean(test_batch_psnr)
    lpips_test = np.mean(test_batch_lpips)

    contents = [
        epoch + 1,
        mse_train,
        psnr_train,
        lpips_train,
        mse_test,
        psnr_test,
        lpips_test,
    ]
    logging_train(contents)

    tqdm.write(format1 % tuple(contents))
    if (epoch + 1) % 100 == 0:
        test_vis_hat = deepjscc(test_vis, N)
        test_vis_hat = test_vis_hat.detach()
        show_and_save(
            r".\%s\images\tets_epoch_%d.png" % (result_dir, epoch + 1),
            make_grid(test_vis_hat.cpu(), 8),
        )

torch.save(deepjscc.state_dict(), r".\%s\weight\deepjscc.pth" % result_dir)


deepjscc.eval()

for i in tqdm(range(0, 21, 2), desc="Test"):
    SNR = i
    N = dB2linear(SNR, P)
    test_batch_loss = []
    test_batch_psnr = []
    test_batch_lpips = []
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
