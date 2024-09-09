import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import datetime, os
from dataloader import dataloader
import argparse
from torchmetrics.image import PeakSignalNoiseRatio
from tqdm.contrib import tenumerate
from models import DeepJSCC

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("channel")
parser.add_argument("snr")
parser.add_argument("c")
args = parser.parse_args()


if args.channel == "SRF":
    ch = "SRF"
elif args.channel == "AWGN":
    ch = "AWGN"
else:
    raise Exception

SNR = int(args.snr)
c = int(args.c)

np.random.seed(42)

results_dir = "results"
datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
result_dir = os.path.join(results_dir, f"{ch}_{datetime}")
os.makedirs(result_dir, exist_ok=True)


epochs = 1
lr_1 = 1e-3
lr_2 = 1e-4


# average power constraint
P = 1
N = P / 10 ** (SNR / 10)
# number of pixels per feature map
PP = 8**2
# image dimension (source bandwidth)
n = 32 * 32 * 3
k = PP * c / 2
k_n = k / n

# load data
train_loader = dataloader(train=True)
test_loader = dataloader(train=False)
deepjscc = DeepJSCC(ch, k, N, c).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(deepjscc.parameters(), lr=lr_1)
PSNR = PeakSignalNoiseRatio(data_range=1.0)


mse_list = []
psnr_list = []
for epoch in range(epochs):
    deepjscc.train()
    train_batch_loss = []
    for i, (x, _) in tenumerate(train_loader):
        x = x.to(device)
        optimizer.zero_grad()
        x_hat = deepjscc(x)
        mse = criterion(x_hat, x)
        mse.backward()
        optimizer.step()
        train_batch_loss.append(mse.item())

    deepjscc.eval()
    test_batch_loss = []
    test_batch_metrics = []
    with torch.no_grad():
        for i, (x, _) in enumerate(test_loader):
            x = x.to(device)
            x_hat = deepjscc(x)
            mse = criterion(x_hat, x)
            psnr = PSNR(x_hat, x)
            test_batch_loss.append(mse.item())
            test_batch_metrics.append(psnr.item())
    mse_avg = np.mean(test_batch_loss)
    psnr_avg = np.mean(test_batch_metrics)
    mse_list.append(mse_avg)
    psnr_list.append(psnr_avg)
    print(f"mse:{mse_avg}, psnr:{psnr_avg}")
