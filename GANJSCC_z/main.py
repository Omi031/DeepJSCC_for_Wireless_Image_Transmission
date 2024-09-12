import datetime, os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, callbacks, optimizers

# from tensorflow.keras.utils import plot_model

import numpy as np
from models import DeepJSCC, Discriminator
import argparse
import lpips
from train import Trainer
from metrics import linear_conversion
from utils import save_image_grid


np.random.seed(42)

# fasing channel
slow_rayleigh_fading = True

if slow_rayleigh_fading:
    ch = "SRF"
else:
    ch = "AWGN"

# results_dir = "results"
# datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# result_dir = os.path.join(results_dir, f"{ch}_{datetime}")
# os.makedirs(result_dir, exist_ok=True)

batch_size = 64

# load data
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
# car
label = 1
train_images = x_train[y_train.flatten() == label]
test_images = x_test[y_test.flatten() == label]

train_images = (train_images - 127.5) / 127.5
test_images = (test_images - 127.5) / 127.5
# train_images = train_images[:100]
# test_images = test_images[:100]
# train_images = train_images.shuffle()
train_dataset = (
    tf.data.Dataset.from_tensor_slices(train_images)
    .shuffle(len(train_images))
    .batch(64)
)
test_dataset = tf.data.Dataset.from_tensor_slices(test_images).batch(64)

true = train_images[:10]
true = linear_conversion(true, val_pre=[-1, 1], val_new=[0, 1])
path = "test.png"
save_image_grid(true, 2, 5, path)

epochs = 1
# Change learning rate to lr_2 from lr_1 after 500k iterations


# parser = argparse.ArgumentParser()
# parser.add_argument("--ch", default="AWGN")
# parser.add_argument("--snr", default=0)
# parser.add_argument("--c", default=4)
# args = parser.parse_args()


# if args.channel == "SRF":
#     ch = "SRF"
# elif args.channel == "AWGN":
#     ch = "AWGN"
# else:
#     raise Exception

# SNR = int(args.snr)
# c = int(args.c)

SNR = 0
c = 8
ch = "AWGN"

# average power constraint
P = 1
N = P / 10 ** (SNR / 10)
# number of pixels per feature map
PP = 8**2
# image dimension (source bandwidth)
n = 32 * 32 * 3
k = PP * c / 2
k_n = k / n
z_dim = 2 * k
print(z_dim)


deepjscc_optim = optimizers.Adam(learning_rate=1e-4)
dis_optim = optimizers.Adam(learning_rate=1e-4)

deepjscc = DeepJSCC(z_dim, k, P, N, ch="AWGN")
discriminator = Discriminator()
trainer = Trainer(deepjscc, discriminator, deepjscc_optim, dis_optim, val=[-1, 1])

for epoch in range(epochs):
    psnr_batch = []
    lpips_batch = []
    deepjscc_loss_batch = []
    dis_loss_batch = []

    for x in train_dataset:
        deepjscc_loss, dis_loss = trainer.train_step(x)

    for x in test_dataset:
        deepjscc_loss, dis_loss, psnr, lpips = trainer.test_step(x)
        deepjscc_loss_batch.append(deepjscc_loss)
        dis_loss_batch.append(dis_loss)
        psnr_batch.append(psnr)
        lpips_batch.append(lpips)

    psnr_avg = np.mean(psnr_batch)
    lpips_avg = np.mean(lpips_batch)
    deepjscc_loss_avg = np.mean(deepjscc_loss_batch)
    dis_loss_avg = np.mean(dis_loss_batch)
    print(
        "Epoch {}, PSNR: {:.2f}, LPIPS: {:.4g}, DeepJSCC Loss: {:.2g}, Dis Loss: {:.2g}, z_dim: {}".format(
            epoch + 1, psnr_avg, lpips_avg, deepjscc_loss_avg, dis_loss_avg, z_dim
        )
    )
    if epoch % 10 == 0:
        true = train_images[:10]
        pred = trainer.predict(true)
        true = linear_conversion(true, val_pre=[-1, 1], val_new=[0, 1])
        pred = linear_conversion(pred, val_pre=[-1, 1], val_new=[0, 1])
        path = "train_epoch{}.png".format(epoch + 1)
        save_image_grid(pred, 2, 5, path)
