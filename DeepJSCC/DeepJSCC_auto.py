import tensorflow as tf
from tensorflow.keras import datasets, layers, models, callbacks
from tensorflow.keras.utils import plot_model
import datetime, os
import Layers, Metrics
import numpy as np
from lpips_tensorflow import lpips_tf

np.random.seed(42)

# fasing channel
slow_rayleigh_fading = False

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
train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0
# train_images = train_images.shuffle()
np.random.shuffle(train_images)
np.random.shuffle(test_images)
train_images = train_images[:50000]
test_images = test_images[:10000]

batch_size = 64
epochs = 1
# Change learning rate to lr_2 from lr_1 after 500k iterations
lr_1 = 1e-3
lr_2 = 1e-4

# SNR[dB]
SNR_list = [0, 10, 20]

x_list = [4, 8, 16, 24, 32, 40, 46]  # AWGN channel
# x_list = [ 8, 16, 32, 48, 64, 80, 92] # Slow Rayleigh fading channel


times = len(SNR_list) * len(x_list)
MSE = [[-1] * len(x_list) for i in range(len(SNR_list))]
PSNR = [[-1] * len(x_list) for i in range(len(SNR_list))]
LPIPS = [[-1] * len(x_list) for i in range(len(SNR_list))]

# average power constraint
P = 1
# number of pixels per feature map
PP = 8**2
# image dimension (source bandwidth)
n = 32 * 32 * 3


def lr_scheduler(epoch, lr):
    iteration = epoch * (len(train_images) // batch_size)
    if iteration >= 500000:
        return lr_2
    return lr


lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

for i, SNR in enumerate(SNR_list):
    # noise power
    N = P / 10 ** (SNR / 10)
    result_file = os.path.join(result_dir, f"{ch}_{SNR}dB_epoch{epochs}_{datetime}.txt")
    with open(result_file, mode="a") as f:
        f.write("k/n, MSE, PSNR")

    for j, x in enumerate(x_list):
        # bandwidth compression ratio
        k_n = PP / (2 * n) * x
        # channel dimension (channel bandwidth)
        k = int(n * k_n)
        # number of filters in the last convolution layer of the encoder
        c = int(2 * k / PP)

        # DeepJSCC model
        model = models.Sequential(name="DeepJSCC")
        # encorder
        model.add(
            layers.Conv2D(
                16,
                (5, 5),
                strides=2,
                padding="same",
                input_shape=(32, 32, 3),
                name="Encoder_Conv2D_1",
            )
        )
        model.add(layers.PReLU(name="Encoder_PReLU_1"))

        model.add(
            layers.Conv2D(
                32, (5, 5), strides=2, padding="same", name="Encoder_Conv2D_2"
            )
        )
        model.add(layers.PReLU(name="Encoder_PReLU_2"))

        model.add(
            layers.Conv2D(
                32, (5, 5), strides=1, padding="same", name="Encoder_Conv2D_3"
            )
        )
        model.add(layers.PReLU(name="Encoder_PReLU_3"))

        model.add(
            layers.Conv2D(
                32, (5, 5), strides=1, padding="same", name="Encoder_Conv2D_4"
            )
        )
        model.add(layers.PReLU(name="Encoder_PReLU_4"))

        model.add(
            layers.Conv2D(c, (5, 5), strides=1, padding="same", name="Encoder_Conv2D_5")
        )
        model.add(layers.PReLU(name="Encoder_PReLU_5"))

        model.add(Layers.Normalization(k, P))
        # model.add(Layers.Modulation())
        # model.add(Layers.Demodulation())

        # add channel noise
        if slow_rayleigh_fading == True:
            model.add(Layers.Slow_Rayleigh_Fading_Channel(N))
        else:
            model.add(Layers.AWGN_Channel(N))

        # encorder
        model.add(
            layers.Conv2DTranspose(
                32, (5, 5), strides=1, padding="same", name="Decoder_TransConv2D_1"
            )
        )
        model.add(layers.PReLU(name="Decoder_PReLU_1"))
        model.add(
            layers.Conv2DTranspose(
                32, (5, 5), strides=1, padding="same", name="Decoder_TransConv2D_2"
            )
        )
        model.add(layers.PReLU(name="Decoder_PReLU_2"))
        model.add(
            layers.Conv2DTranspose(
                32, (5, 5), strides=1, padding="same", name="Decoder_TransConv2D_3"
            )
        )
        model.add(layers.PReLU(name="Decoder_PReLU_3"))
        model.add(
            layers.Conv2DTranspose(
                16, (5, 5), strides=2, padding="same", name="Decoder_TransConv2D_4"
            )
        )
        model.add(layers.PReLU(name="Decoder_PReLU_4"))
        model.add(
            layers.Conv2DTranspose(
                3,
                (5, 5),
                strides=2,
                padding="same",
                activation="sigmoid",
                name="Decoder_TransConv2D_5",
            )
        )

        # model.summary()
        # plot_model(model, show_shapes=True)

        file_name = os.path.join(
            result_dir, f"{ch}_{SNR}dB_k_n{round(k_n, 2)}_epoch{epochs}_{datetime}"
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_1),
            loss=tf.keras.losses.MSE,
            metrics=[Metrics.PSNR],
        )

        # log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # os.makedirs(log_dir, exist_ok=True)
        csv_logger = callbacks.CSVLogger(f"{file_name}.log")
        # tensorboard_callback = callbacks.TensorBoard(log_dir='logs', histogram_freq=0)

        # train
        print(f"{len(x_list)*i+j+1}/{times}：SNR={SNR}dB, k/n={round(k_n, 2)}")
        model.fit(
            train_images,
            train_images,
            validation_data=[test_images, test_images],
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[csv_logger, lr_callback],
            # callbacks=[tensorboard_callback]
        )

        model.save(f"{file_name}.keras")

        print(f"{len(x_list)*i+j+1}/{times}：SNR={SNR}dB, k/n={round(k_n, 2)}")
        mse = 0
        psnr = 0
        lpips = 0
        for k in range(10):
            m, p, l = model.evaluate(test_images, test_images, batch_size=64)
            mse += m
            psnr += p
            lpips += l
        MSE[i][j] = float(mse / 10)
        PSNR[i][j] = float(psnr / 10)
        LPIPS[i][j] = float(lpips / 10)
        print(f"MSE:{MSE[i][j]}, PSNR:{PSNR[i][j]}, LPIPS:{LPIPS[i][j]}")

        with open(result_file, mode="a") as f:
            k_n = round(PP / (2 * n) * x, 2)
            f.write(f"\n{k_n}, {MSE[i][j]}, {PSNR[i][j]}, {LPIPS[i][j]}")


print("======result======")
for i, SNR in enumerate(SNR_list):
    print()
    print(f"SNR={SNR}dB")
    print("k/n, MSE, PSNR, LPIPS")
    for j, x in enumerate(x_list):
        k_n = PP / (2 * n) * x
        print(round(k_n, 2), MSE[i][j], PSNR[i][j], LPIPS[i][j])
print("==================")

for i, SNR in enumerate(SNR_list):
    print()
    print(f"SNR={SNR}dB")
    print("MSE")
    for M in MSE[i]:
        print(M)
    print("PSNR")
    for P in PSNR[i]:
        print(P)
    print("LPIPS")
    for L in LPIPS[i]:
        print(L)
