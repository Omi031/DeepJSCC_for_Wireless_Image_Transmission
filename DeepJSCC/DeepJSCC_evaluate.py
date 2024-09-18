import cv2
import Layers
import matplotlib.pyplot as plt
import Metrics
import numpy as np
import tensorflow as tf
from lpips_tensorflow import lpips_tf
from tensorflow.keras import datasets, models
from tqdm import tqdm

MSE = []
PSNR = []

# SNR[dB]
SNR = 0
# average power
P = 1
# noise power
N = P / 10 ** (SNR / 10)
x = 8
# patch^2
PP = 8**2
# image dimension (source bandwidth)
n = 32 * 32 * 3
# bandwidth compression ratio
k_n = PP / (2 * n) * x
# channel dimension (channel bandwidth)
k = int(n * k_n)
# number of filters in the last convolution layer of the encoder
c = int(2 * k / PP)


MODEL = (
    r".\results\AWGN_20240826-165218\AWGN_0dB_k_n0.04_epoch1000_20240826-165218.keras"
)

# load model
custom_objects = {
    "Normalization": Layers.Normalization,
    "AWGN_Channel": Layers.AWGN_Channel,
    "Slow_Rayleigh_Fading_Channel": Layers.Slow_Rayleigh_Fading_Channel,
    "PSNR": Metrics.PSNR,
    # "N": N,
}
model = models.load_model(MODEL, custom_objects=custom_objects)

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=[Metrics.PSNR, lpips_tf.lpips],
)

# load image
(_, _), (imgs, _) = datasets.cifar10.load_data()

imgs = imgs[:10000]

# for i, img in enumerate(imgs):
#   cv2.imwrite(f'original_{i}.png', img)

imgs_norm = imgs.astype("float32") / 255.0

mse = []
psnr = []
lpips = []
for i in tqdm(range(10)):
    m, p, l = model.evaluate(imgs_norm, imgs_norm, batch_size=64)
    mse.append(m)
    psnr.append(p)
    lpips.append(l)
mse_avg = np.mean(mse)
psnr_avg = np.mean(psnr)
lpips_avg = np.mean(lpips)

print("MSE: %f\tPSNR: %f\tLPIPS: %f" % (mse_avg, psnr_avg, lpips_avg))

imgs_pred_norm = model.predict(imgs_norm)

imgs_pred = imgs_pred_norm * 255

# for i, img in enumerate(imgs_pred):
#   cv2.imwrite(f'predict_{i}.png', img)
