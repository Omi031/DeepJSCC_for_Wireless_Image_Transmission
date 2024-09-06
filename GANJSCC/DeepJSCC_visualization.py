import tensorflow as tf
from tensorflow.keras import datasets, models
import matplotlib.pyplot as plt
import cv2
import Layers, Metrics

MSE = []
PSNR = []

# SNR[dB]
SNR = 20
# average power
P = 1
# noise power
N = P/10**(SNR/10)
x = 40
# patch^2
PP = 8**2
# image dimension (source bandwidth)
n = 32*32*3
# bandwidth compression ratio
k_n = PP/(2*n)*x
# channel dimension (channel bandwidth)
k = int(n*k_n)
# number of filters in the last convolution layer of the encoder
c = int(2*k/PP)


MODEL = 'model_SRayleigh_20dB_k_n0.42_x40_epoch100.keras'

# load model
custom_objects={'Normalization': Layers.Normalization,
                'AWGN_Channel': Layers.AWGN_Channel,
                'Slow_Rayleigh_Fading_Channel': Layers.Slow_Rayleigh_Fading_Channel,
                'PSNR': Metrics.PSNR}
model = models.load_model(MODEL, custom_objects=custom_objects)

model.compile(optimizer='adam', loss='mse', metrics=[Metrics.PSNR])

# load image
(_,_), (imgs,_) = datasets.cifar10.load_data()

imgs = imgs[:]

for i, img in enumerate(imgs):
  cv2.imwrite(f'original_{i}.png', img)

imgs_norm = imgs.astype('float32') / 255.0

m, p = model.evaluate(imgs_norm, imgs_norm, batch_size=64)
print(m,p)

imgs_pred_norm = model.predict(imgs_norm)

imgs_pred = imgs_pred_norm*255

for i, img in enumerate(imgs_pred):
  cv2.imwrite(f'predict_{i}.png', img)