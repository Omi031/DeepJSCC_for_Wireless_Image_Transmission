import cv2, tqdm
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio

# 任意のqualityにおけるJPEG2000のデータサイズを取得
def get_jpeg_size(img, quality):
  jpeg_bytes = cv2.imencode('.jp2', img, [int(cv2.IMWRITE_JPEG2000_COMPRESSION_X1000), quality])[1].tobytes()
  data_size = len(jpeg_bytes)
  return data_size

# 任意のqualityでJPEG2000圧縮
def jpeg_compression(img, quality):
  img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
  _, jpeg = cv2.imencode('.jp2', img, (cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, quality))
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
    dest[:,:,i] = np.full(img_shape[:2], mean)
  return dest

# 処理部
(_, _), (org_img, _) = tf.keras.datasets.cifar10.load_data()
org_img = org_img[:100]

psnr_avg_list = []
n = org_img[0].size
k_n_list = [0.04, 0.08, 0.16, 0.25, 0.33, 0.42, 0.48]
SNR_dB = 10
SNR = 10**(SNR_dB/10)
C = np.log2(1+SNR)
# quality = np.arange(0, 1001, 1)

for k_n in k_n_list:
  R_max = k_n*C
  data_size_max = n*R_max/8
  psnr = []

  for img in tqdm.tqdm(org_img):
    data_size_old = 0
    for Q in range(0, 1001, 10):
      data_size_new = get_jpeg_size(img, Q)
      if data_size_new > data_size_max:
        if Q == 0:
          jpeg = img_destruction(img)
        else:
          for q in range(Q, Q+10, 1):
            data_size_new = get_jpeg_size(img, Q)
            if data_size_new > data_size_max:
              jpeg = jpeg_compression(img, q-1)
              break
            data_size_old = data_size_new
          break
      data_size_old = data_size_new
    p = peak_signal_noise_ratio(img, jpeg, data_range=255)
    if p != float('inf'):
      psnr.append(p)

  psnr_avg = np.mean(psnr)
  psnr_avg_list.append(psnr_avg)

  print(f'k/n={k_n}, data_size={data_size_max}Bytes, PSNR={psnr_avg}, inf:{len(org_img)-len(psnr)}')

# 結果表示
k_n_list_c = [0.04, 0.08, 0.16, 0.25, 0.33, 0.42, 0.48]
if SNR_dB == 0:
  psnr_avg_list_c = [14, 14, 14, 14, 14, 14, 14]
elif SNR_dB == 10:
  psnr_avg_list_c = [14, 14, 17, 26, 30, 33, 35]
elif SNR_dB == 20:
  psnr_avg_list_c = [14, 16, 30, 35, 41, 41, 46]
else:
  psnr_avg_list_c = np.zeros(k_n_list_c.size)


fig = plt.figure(figsize=(7,5))
axes = fig.add_subplot(1,1,1, title=f'JPEG2000 SNR={SNR_dB}dB', xlabel='k/n', ylabel='PSNR(dB)')

axes.plot(k_n_list_c, psnr_avg_list_c, '-', marker='o')
axes.plot(k_n_list, psnr_avg_list, '-', marker='o')

axes.legend(['論文値', '検証値'], prop={"family":"MS Gothic"})
axes.set_ylim([10,50])

plt.grid()
plt.show()