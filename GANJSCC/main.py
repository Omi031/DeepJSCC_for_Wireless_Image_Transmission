import tensorflow as tf
from tensorflow.keras import datasets, layers, models, callbacks, optimizers
from tensorflow.keras.utils import plot_model
import datetime, os
from models import (
    DeepJSCC,
    Discriminator,
    Slow_Rayleigh_Fading_Channel,
    AWGN_Channel,
    Normalization,
)
import numpy as np
from tqdm import tqdm
from metrics import PSNR, LPIPS, tf_tensor2pt_tensor


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

# load data
(train_images, _), (test_images, _) = datasets.cifar10.load_data()
train_images = (train_images - 127.5) / 127.5
test_images = (test_images - 127.5) / 127.5
train_images = train_images[:100]
test_images = test_images[:100]
# train_images = train_images.shuffle()
train_dataset = (
    tf.data.Dataset.from_tensor_slices(train_images)
    .shuffle(len(train_images))
    .batch(64)
)
test_dataset = tf.data.Dataset.from_tensor_slices(test_images).batch(64)


batch_size = 64

# Change learning rate to lr_2 from lr_1 after 500k iterations
lr_1 = 1e-3
lr_2 = 1e-4


def lr_scheduler(epoch, lr):
    iteration = epoch * (len(train_images) // batch_size)
    if iteration >= 500000:
        return lr_2
    return lr


lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real, fake):
    real_loss = cross_entropy(tf.ones_like(real), real)
    fake_loss = cross_entropy(tf.zeros_like(fake), fake)
    loss = real_loss + fake_loss
    return loss


def deepjscc_loss(fake):
    return cross_entropy(tf.ones_like(fake), fake)


djscc_optim = optimizers.Adam(1e-3)
dis_optim = optimizers.Adam(1e-3)


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

epochs = 1


def deepjscc(c, k, P, N, slow_rayleigh_fading=False):
    if slow_rayleigh_fading == True:
        channel = Slow_Rayleigh_Fading_Channel(N)
    else:
        channel = AWGN_Channel(N)
    model = models.Sequential(name="DeepJSCC")
    # encorder
    model.add(
        layers.Conv2D(16, (5, 5), strides=2, padding="same", input_shape=(32, 32, 3))
    )
    model.add(layers.PReLU())

    model.add(layers.Conv2D(32, (5, 5), strides=2, padding="same"))
    model.add(layers.PReLU())

    model.add(layers.Conv2D(32, (5, 5), strides=1, padding="same"))
    model.add(layers.PReLU())

    model.add(layers.Conv2D(32, (5, 5), strides=1, padding="same"))
    model.add(layers.PReLU())

    model.add(layers.Conv2D(c, (5, 5), strides=1, padding="same"))
    model.add(layers.PReLU())

    model.add(Normalization(k, P))

    # add channel noise
    model.add(channel)

    # encorder
    model.add(layers.Conv2DTranspose(32, (5, 5), strides=1, padding="same"))
    model.add(layers.PReLU())
    model.add(layers.Conv2DTranspose(32, (5, 5), strides=1, padding="same"))
    model.add(layers.PReLU())
    model.add(layers.Conv2DTranspose(32, (5, 5), strides=1, padding="same"))
    model.add(layers.PReLU())
    model.add(layers.Conv2DTranspose(16, (5, 5), strides=2, padding="same"))
    model.add(layers.PReLU())
    model.add(
        layers.Conv2DTranspose(
            3,
            (5, 5),
            strides=2,
            padding="same",
            activation="sigmoid",
        )
    )
    return model


def discriminator():
    model = models.Sequential(name="Discriminator")

    model.add(layers.Conv2D(16, (5, 5), strides=1, padding="same"))
    model.add(layers.PReLU())
    model.add(layers.Conv2D(32, (5, 5), strides=2, padding="same"))
    model.add(layers.PReLU())
    model.add(layers.Conv2D(64, (5, 5), strides=2, padding="same"))
    model.add(layers.PReLU())
    model.add(layers.Conv2D(128, (5, 5), strides=2, padding="same"))
    model.add(layers.PReLU())
    model.add(layers.Dense(512))
    model.add(layers.PReLU())
    model.add(layers.Dense(1))
    model.add(layers.Activation("sigmoid"))
    return model


deepjscc = deepjscc(c, k, P, N)

discriminator = discriminator()


@tf.function
def train_step(x):
    with tf.GradientTape() as djscc_tape, tf.GradientTape() as dis_tape:
        x_hat = deepjscc(x)

        x_dis = discriminator(x)
        x_hat_dis = discriminator(x_hat)

        djscc_loss = deepjscc_loss(x_hat_dis)
        dis_loss = discriminator_loss(x_dis, x_hat_dis)

    djscc_grads = djscc_tape.gradient(djscc_loss, deepjscc.trainable_variables)
    dis_grads = dis_tape.gradient(dis_loss, discriminator.trainable_variables)
    djscc_optim.apply_gradients(zip(djscc_grads, deepjscc.trainable_variables))
    dis_optim.apply_gradients(zip(dis_grads, discriminator.trainable_variables))
    return djscc_loss, dis_loss


psnr_fn = PSNR()
lpips_fn = LPIPS(val=[0, 1])


def test_step(x):
    x_hat = deepjscc(x)
    x_dis = discriminator(x)
    x_hat_dis = discriminator(x_hat)
    djscc_loss = deepjscc_loss(x_hat_dis)
    dis_loss = discriminator_loss(x_dis, x_hat_dis)
    psnr = tf.reduce_mean(psnr_fn(x, x_hat))
    x = tf_tensor2pt_tensor(x)
    x_hat = tf_tensor2pt_tensor(x_hat)
    lpips = lpips_fn(x, x_hat).detach().numpy()
    lpips = tf.reduce_mean(lpips)
    return djscc_loss, dis_loss, psnr, lpips


for epoch in range(epochs):
    train_djscc_loss_batch = []
    train_dis_loss_batch = []
    test_djscc_loss_batch = []
    test_dis_loss_batch = []
    test_psnr_batch = []
    test_lpips_batch = []

    for x in train_dataset:
        djscc_loss, dis_loss = train_step(x)
        train_djscc_loss_batch.append(djscc_loss)
        train_dis_loss_batch.append(dis_loss)

    for x in test_dataset:
        djscc_loss, dis_loss, psnr, lpips = test_step(x)
        test_djscc_loss_batch.append(djscc_loss)
        test_dis_loss_batch.append(dis_loss)
        test_psnr_batch.append(psnr)
        test_lpips_batch.append(lpips)

    train_djscc_loss = np.mean(train_djscc_loss_batch)
    train_dis_loss = np.mean(train_dis_loss_batch)
    test_djscc_loss = np.mean(test_djscc_loss_batch)
    test_dis_loss = np.mean(test_dis_loss_batch)
    test_psnr = np.mean(test_psnr_batch)
    test_lpips = np.mean(test_lpips_batch)

    print(
        train_djscc_loss,
        train_dis_loss,
        test_djscc_loss,
        test_dis_loss,
        test_psnr,
        test_lpips,
    )
