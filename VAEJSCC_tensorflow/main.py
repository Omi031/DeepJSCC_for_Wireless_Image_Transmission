import tensorflow as tf
from tensorflow.keras import datasets, layers, models, callbacks, optimizers
from tensorflow.keras.utils import plot_model
import datetime, os
import numpy as np
from models import VAE
import argparse
import lpips

np.random.seed(42)

# fasing channel
slow_rayleigh_fading = True

if slow_rayleigh_fading:
    ch = "SRF"
else:
    ch = "AWGN"

results_dir = "results"
datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
result_dir = os.path.join(results_dir, f"{ch}_{datetime}")
os.makedirs(result_dir, exist_ok=True)

batch_size = 64

# load data
(train_images, _), (test_images, _) = datasets.cifar10.load_data()
train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0
train_images = train_images[:100]
test_images = test_images[:100]
# train_images = train_images.shuffle()
np.random.shuffle(train_images)
train_images = tf.data.Dataset.from_tensor_slices(train_images)
train_images = train_images.batch(batch_size)
test_images = tf.data.Dataset.from_tensor_slices(test_images)
test_images = test_images.batch(batch_size)
# train_images = train_images[:50000]
# test_images = test_images[:10000]

epochs = 1
# Change learning rate to lr_2 from lr_1 after 500k iterations
lr_1 = 1e-3
lr_2 = 1e-4


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


def lr_scheduler(epoch, lr):
    iteration = epoch * (len(train_images) // batch_size)
    if iteration >= 500000:
        return lr_2
    return lr


lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

optimizer = optimizers.Adam(learning_rate=1e-3)

model = VAE(z_dim, N, optimizer=optimizer, ch="AWGN")
LPIPS = lpips.LPIPS(net="vgg")


for epoch in range(epochs):
    psnr_batch = []
    lpips_batch = []
    rc_loss_batch = []
    kl_loss_batch = []
    loss_batch = []
    for x in train_images:
        rc_loss, kl_loss, loss = model.train_step(x)
    for x in test_images:
        x_hat, rc_loss, kl_loss, loss = model.test_step(x)
        psnr_batch.append(tf.reduce_mean(tf.image.psnr(x, x_hat, max_val=1.0)))
        # x_lpips = tf.transpose(2 * x - 1, perm=[0, 3, 1, 2])
        # x_hat_lpips = tf.transpose(2 * x_hat - 1, perm=[0, 3, 1, 2])
        # lpips_batch.append(LPIPS(x_lpips, x_hat_lpips))
        rc_loss_batch.append(rc_loss)
        kl_loss_batch.append(kl_loss)
        loss_batch.append(loss)

    psnr_avg = np.mean(psnr_batch)
    lpips_avg = np.mean(lpips_batch)
    rc_loss_avg = np.mean(rc_loss_batch)
    kl_loss_avg = np.mean(kl_loss_batch)
    loss_avg = np.mean(loss_batch)
    print(
        "Epoch {}, PSNR: {:.2f}, LPIPS: {:.4g}, RC Loss: {:.2g}, KL Loss: {:.2g}, Loss: {:.2g}".format(
            epoch + 1,
            psnr_avg,
            lpips_avg,
            rc_loss_avg,
            kl_loss_avg,
            loss_avg,
        )
    )
