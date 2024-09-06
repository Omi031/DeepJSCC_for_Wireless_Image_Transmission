import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import datetime, os
from dataloader import dataloader
import argparse
from models import DeepJSCC

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("channel")
parser.add_argument("snr")
parser.add_argument("c")
args = parser.parse_args()


if args.channel != "SRF":
    ch = "SRF"
elif args.channel == "AWGN":
    ch = "AWGN"
else:
    raise Exception

SNR = args.snr
c = args.c

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
# number of pixels per feature map
PP = 8**2
# image dimension (source bandwidth)
n = 32 * 32 * 3
k = PP * c / 2
k_n = k / n

# load data
train_loader = dataloader(train=True)
test_loader = dataloader(train=False)
deepjscc = DeepJSCC().to(device)

for epoch in range(epochs):
    for i, (train_imgs, _) in enumerate(train_loader):
        
