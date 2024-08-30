import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import Layers
import Metrics

# load data
(train_images, _), (test_images, _) = datasets.cifar10.load_data()
train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0

train_images = train_images[:50000]
test_images = test_images[:10000]

MSE = []
PSNR = []

# train epoch
epochs = 100

# SNR[dB]
SNR = 0
# average power
P = 1
# noise power
N = P / 10 ** (SNR / 10)
x_list = [5, 10, 15, 20, 30, 40]
# patch^2
PP = 8**2
# image dimension (source bandwidth)
n = 32 * 32 * 3
# bandwidth compression ratio
k_n_list = [PP / (2 * n) * x for x in x_list]
# channel dimension (channel bandwidth)
k_list = [int(n * k_n) for k_n in k_n_list]
# number of filters in the last convolution layer of the encoder
c_list = [int(2 * k / PP) for k in k_list]

for i in range(len(x_list)):
    print(f"{i+1}/{len(x_list)}：SNR={SNR}dB, k/n={round(k_n_list[i], 2)}")
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
        layers.Conv2D(32, (5, 5), strides=2, padding="same", name="Encoder_Conv2D_2")
    )
    model.add(layers.PReLU(name="Encoder_PReLU_2"))

    model.add(
        layers.Conv2D(32, (5, 5), strides=1, padding="same", name="Encoder_Conv2D_3")
    )
    model.add(layers.PReLU(name="Encoder_PReLU_3"))

    model.add(
        layers.Conv2D(32, (5, 5), strides=1, padding="same", name="Encoder_Conv2D_4")
    )
    model.add(layers.PReLU(name="Encoder_PReLU_4"))

    model.add(
        layers.Conv2D(
            c_list[i], (5, 5), strides=1, padding="same", name="Encoder_Conv2D_5"
        )
    )
    model.add(layers.PReLU(name="Encoder_PReLU_5"))

    model.add(Layers.Normalization(k_list[i], P))

    # add channel noise (AWGN)
    # model.add(layers.GaussianNoise(N, name='Channel_AWGN'))
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

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=10 ** (-3)),
        loss=tf.keras.losses.MSE,
        metrics=[Metrics.PSNR],
    )

    # train
    model.fit(
        train_images,
        train_images,
        validation_data=[test_images, test_images],
        epochs=epochs,
        batch_size=64,
    )

    model.save(f"model_{str(SNR)}dB_k_n{str(round(k_n_list[i], 2))}_epoch{epochs}.h5")

    mse = 0
    psnr = 0
    for j in range(10):
        m, p = model.evaluate(test_images, test_images, batch_size=64)
        mse += m
        psnr += p
    MSE.append(mse / 10)
    PSNR.append(psnr / 10)
    print(f"MSE:{MSE[i]}, PSNR:{PSNR[i]}")


print("result")
print(f"SNR={SNR}dB")
print("k/n, MSE, PSNR")
for i in range(len(MSE)):
    print(round(k_n_list[i], 2), MSE[i], PSNR[i])

for M in MSE:
    print(MSE)
for P in PSNR:
    print(P)
