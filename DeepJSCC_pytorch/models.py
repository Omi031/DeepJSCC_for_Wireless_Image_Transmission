import torch
import torch.nn as nn
from torchinfo import summary
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, c):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.conv5 = nn.Conv2d(32, c, 5, stride=1, padding=2)
        self.relu = nn.PReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        return x


class Decoder(nn.Module):
    def __init__(self, c):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(c, 32, 5, stride=1, padding=2)
        self.deconv2 = nn.ConvTranspose2d(32, 32, 5, stride=1, padding=2)
        self.deconv3 = nn.ConvTranspose2d(32, 32, 5, stride=1, padding=2)
        self.deconv4 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1)
        self.relu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.relu(self.deconv3(x))
        x = self.relu(self.deconv4(x))
        x = self.sigmoid(self.deconv5(x))
        return x


# class Channel:
#     def __init__(self, k, N):
#         self.k = k
#         self.std = np.sqrt(N)

#     def awgn(self, x):
#         noise = torch.normal(mean=0, std=self.std, size=(x.size))
#         x = x + noise
#         return x

#     def slow_raileigh_fading(self, x):
#         return x


class Normalization(nn.Module):
    def __init__(self, k, P):
        super(Normalization, self).__init__()
        self.k = k
        self.P = P

    def forward(self, x):
        x_norm = torch.sqrt(torch.sum(torch.square(x), dim=[1, 2, 3], keepdim=True))
        p = np.sqrt(self.k * self.P)
        x = p * x / x_norm
        return x


class AWGN(nn.Module):
    def __init__(self, k):
        super(AWGN, self).__init__()
        self.k = k

    def forward(self, x, N):
        std = float(np.sqrt(N / 2))
        noise = torch.normal(mean=0, std=std, size=(x.size()))
        x = x + noise
        return x


class DeepJSCC(nn.Module):
    def __init__(self, ch, k, P, c):
        super(DeepJSCC, self).__init__()
        self.encoder = Encoder(c)
        self.decoder = Decoder(c)
        self.norm = Normalization(k, P)
        if ch == "AWGN":
            self.channel = AWGN(k)
        elif ch == "SRF":
            # self.channel = Channel(k, N).slow_raileigh_fading()
            pass

    def forward(self, x, N):
        x = self.encoder(x)
        x = self.norm(x)
        x = self.channel(x, N)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    ch = "AWGN"
    c = 4
    PP = 8**2
    k = PP * c / 2
    N = 1
    P = 1
    encoder = Encoder(c)
    decoder = Decoder(c)
    # channel = AWGN(k, N)
    deepjscc = DeepJSCC(ch, k, P, c)
    print(summary(model=deepjscc, input_size=(64, 3, 64, 64)))
