import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio

# 任意のqualityにおけるJPEGのデータサイズを取得
def get_jpeg_size(img, quality):
  jpeg_bytes = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])[1].tobytes()
  data_size = len(jpeg_bytes)
  return data_size

# 任意のqualityでJPEG圧縮
def jpeg_compression(img, quality):
  img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
  _, jpeg = cv2.imencode('.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, quality))
  jpeg = cv2.imdecode(jpeg, cv2.IMREAD_UNCHANGED)
  jpeg = cv2.cvtColor(jpeg, cv2.COLOR_BGR2RGB)
  return jpeg

# 各色チャネルで画素値を平均
def img_destruction(img):
  img_shape = img.shape
  dest = np.zeros(img_shape)
  mean = np.zeros(img_shape[2])
  for i in range(img_shape[2]):
    mean = np.mean(img[:,:,i])
    dest[:,:,i] = np.full(img_shape[:2], int(mean))
  dest = np.clip(dest, 0, 255).astype(np.uint8)
  return dest

# 処理部
(_, _), (org_img, _) = tf.keras.datasets.cifar10.load_data()
org_img = org_img[0]

psnr_list = []
jpeg_list = []
quality_list = []
n = org_img.size
k_n_list = [0.04, 0.08, 0.16, 0.25, 0.33, 0.42, 0.48]
SNR_dB = 20
SNR = 10**(SNR_dB/10)
C = np.log2(1+SNR)
quality = np.arange(0, 101, 1)

for k_n in k_n_list:
  R_max = k_n*C
  data_size_max = n*R_max/8
  psnr = []

  data_size_old = 0
  for q in quality:
    data_size_new = get_jpeg_size(org_img, q)      
    if data_size_new > data_size_max:
      if q == 0:
        jpeg = img_destruction(org_img)
        quality_list.append(q)
      else:
        jpeg = jpeg_compression(org_img, q-1)
        quality_list.append(q-1)
      jpeg_list.append(jpeg)
      break
    data_size_old = data_size_new
  psnr = peak_signal_noise_ratio(org_img, jpeg, data_range=255)
    
  psnr_list.append(psnr)

  print(f'k/n={k_n}, data_size={data_size_max}Bytes, PSNR={psnr}')

# 結果表示
fig, ax = plt.subplots(2, 4, figsize=(20, 10))
ax[0,0].imshow(org_img)
ax[0,0].set_title('Original Image')
ax[0,0].axis('off')

pos = [[0,1],[0,2],[0,3],[1,0],[1,1],[1,2],[1,3]]
for i in range(len(k_n_list)):
  ax[pos[i][0],pos[i][1]].imshow(jpeg_list[i])
  ax[pos[i][0],pos[i][1]].set_title(f'JPEG k/n={k_n_list[i]}\nquality={quality_list[i]} PSNR={round(psnr_list[i],2)}dB')
  ax[pos[i][0],pos[i][1]].axis('off')

plt.show()