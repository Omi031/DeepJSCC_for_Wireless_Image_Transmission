import torch
import torch.nn as nn
from torchinfo import summary
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, k):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.PReLU()
        self.conv2 = nn.Conv2d(16, 32, 5, stride=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 5, stride=1, padding=2, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 5, stride=1, padding=2, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        # self.conv5 = nn.Conv2d(128, 256, 5, stride=1, padding=2)
        # self.bn5 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(128 * 8 * 8, 4096, bias=False)
        self.bn6 = nn.BatchNorm1d(4096)
        self.fc_mean = nn.Linear(4096, 2 * int(k))
        self.fc_logvar = nn.Linear(4096, 2 * int(k))

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        # x = self.relu(self.bn5(self.conv5(x)))
        x = x.view(x.size()[0], -1)
        x = self.relu(self.bn6(self.fc1(x)))
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, k):
        super(Decoder, self).__init__()
        # self.fc1 = nn.Linear(2 * int(k), 8 * 8 * 256)
        self.fc1 = nn.Linear(2 * int(k), 8 * 8 * 128, bias=False)
        # self.bn1 = nn.BatchNorm1d(8 * 8 * 256)
        self.bn1 = nn.BatchNorm1d(8 * 8 * 128)
        self.relu = nn.PReLU()
        # self.deconv1 = nn.ConvTranspose2d(256, 128, 5, stride=1, padding=2)
        # self.bn2 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=1, padding=2, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 5, stride=1, padding=2, bias=False)
        self.bn4 = nn.BatchNorm2d(32)
        self.deconv4 = nn.ConvTranspose2d(
            32, 16, 5, stride=2, padding=2, output_padding=1, bias=False
        )
        self.bn5 = nn.BatchNorm2d(16)
        self.deconv5 = nn.ConvTranspose2d(
            16, 3, 5, stride=2, padding=2, output_padding=1, bias=False
        )
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = x.view(-1, 128, 8, 8)
        # x = self.relu(self.bn2(self.deconv1(x)))
        x = self.relu(self.bn3(self.deconv2(x)))
        x = self.relu(self.bn4(self.deconv3(x)))
        x = self.relu(self.bn5(self.deconv4(x)))
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
        x_norm = torch.sqrt(torch.sum(torch.square(x), dim=[1], keepdim=True))
        p = np.sqrt(self.k * self.P)
        x = p * x / x_norm
        return x


class AWGN(nn.Module):
    def __init__(self, k, N):
        super(AWGN, self).__init__()
        self.k = k
        self.std = float(np.sqrt(N / 2))

    def forward(self, x):
        noise = torch.normal(mean=0, std=self.std, size=(x.size())).to(device)
        x = x + noise
        return x


class DeepJSCC(nn.Module):
    def __init__(self, ch, k, P, N):
        super(DeepJSCC, self).__init__()
        self.encoder = Encoder(k)
        self.decoder = Decoder(k)
        self.norm = Normalization(k, P)
        if ch == "AWGN":
            self.channel = AWGN(k, N)
        elif ch == "SRF":
            # self.channel = Channel(k, N).slow_raileigh_fading()
            pass

    def forward(self, x, train=True):
        z_mean, z_logvar = self.encoder(x)
        z_mean = self.norm(z_mean)
        if train == True:
            std = z_logvar.mul(0.5).exp_()
            e = torch.randn(std.size()[0], std.size()[1]).to(device)
            z = z_mean + std * e
        else:
            z = z_mean
            z = self.channel(z_mean)
        x = self.decoder(z)
        return x, z_mean, z_logvar


if __name__ == "__main__":
    ch = "AWGN"
    c = 4
    PP = 8**2
    k = PP * c / 2
    N = 1
    encoder = Encoder(c)
    decoder = Decoder(c)
    channel = AWGN(k, N)
    deepjscc = DeepJSCC(ch, k, N, c)
    print(summary(model=deepjscc, input_size=(64, 3, 32, 32)))
