import lpips
import numpy as np
import tensorflow as tf
import torch
from dataloader import dataloader
from lpips_tensorflow import lpips_tf
from tensorflow.keras import datasets
import tensorflow_datasets as tfds
from torchmetrics.image import PeakSignalNoiseRatio

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

(train_images, _), (test_images, _) = datasets.cifar10.load_data()
train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0

train_tensorflow = train_images[:2]
test_tensorflow = test_images[:2]

train_loader = dataloader(train=True, batch_size=2)
test_loader = dataloader(train=False, batch_size=2)

train_batch, _ = next(iter(train_loader))
test_batch, _ = next(iter(test_loader))
train_pytorch = train_batch
test_pytorch = test_batch

# lpips_tensorflow = lpips_tf.lpips(train_tensorflow, test_tensorflow)
# lpips_fn = lpips.LPIPS(net="vgg")
# lpips_pytorch = lpips_fn(train_pytorch, test_pytorch)
# # print(tfds.as_numpy(lpips_tensorflow))
# print(lpips_pytorch.item())

psnr_fn = PeakSignalNoiseRatio(data_range=1, reduction="none")

psnr_tensorflow = tf.image.psnr(train_tensorflow, test_tensorflow, max_val=1.0)
psnr_pytorch = psnr_fn(train_pytorch, test_pytorch)
print(tfds.as_numpy(psnr_tensorflow))
print(psnr_pytorch)
