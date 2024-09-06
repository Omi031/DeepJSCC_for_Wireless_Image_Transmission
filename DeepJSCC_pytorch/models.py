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
        self.conv3 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.conv5 = nn.Conv2d(32, c, 5, stride=1, padding=1)
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
        self.deconv1 = nn.ConvTranspose2d(c, 32, 5, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 32, 5, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 32, 5, stride=1, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, 16, 5, stride=2, padding=2)
        self.deconv5 = nn.ConvTranspose2d(16, 3, 5, stride=2, padding=2)
        self.relu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.relu(self.deconv3(x))
        x = self.relu(self.deconv4(x))
        x = self.sigmoid(self.deconv5(x))
        return x


class Channel:
    def __init__(self, k, N):
        self.k = k
        self.std = np.sqrt(N)

    def awgn(self, x):
        noise = torch.normal(mean=0, std=self.std, size=(x.size))
        x = x + noise
        return x

    def slow_raileigh_fading(self, x):
        return x


class DeepJSCC(nn.Module):
    def __init__(self):
        super(DeepJSCC, self).__init__()
        self.encoder = Encoder()
