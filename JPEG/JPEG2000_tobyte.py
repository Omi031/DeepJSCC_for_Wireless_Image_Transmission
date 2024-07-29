import cv2, tqdm
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio

# 任意のqualityにおけるJPEG2000のデータサイズを取得
def get_jpeg_size(img, quality):
  jpeg_bytes = cv2.imencode('.jp2', img, [int(cv2.IMWRITE_JPEG2000_COMPRESSION_X1000), quality])[1].tobytes()
  with open('jpeg2000_byte_.txt', 'a') as f:
    f.write(str(jpeg_bytes)+'\n')
  





(_, _), (org_img, _) = tf.keras.datasets.cifar10.load_data()
img = org_img[:100]

# for i in range(1001):
#   get_jpeg_size(img[0], i)


for i in range(100):
  get_jpeg_size(img[i], 500)