import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid, save_image

matplotlib.use("Agg")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dB2linear(snr, p):
    return p / 10 ** (snr / 10)


def show_and_save(file_name, img):
    npimg = np.transpose(img.numpy(), (1, 2, 0))
    fig = plt.figure(dpi=200)
    fig.suptitle(file_name, fontsize=14, fontweight="bold")
    plt.imshow(npimg)
    plt.imsave(file_name, npimg)
    plt.close()


def plot_loss(loss_list):
    plt.figure(figsize=(10, 5))
    plt.title("Loss During Training")
    plt.plot(loss_list, label="Loss")

    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


class Logging:
    def __init__(self, dir="", name="log", mode="a", info="", heads=[]):
        file = "%s.txt" % name
        self.path = os.path.join(dir, file)
        self.mode = mode
        if os.path.isfile(self.path):
            print(f"{file} already exists.")
            yn = input("Do you want to override %s? (Y/N):" % file)
            if yn == "y" or yn == "Y":
                os.remove(self.path)
            else:
                raise Exception(f"Please delete the file: {self.path}")

        info = "\n".join(f"{i}" for i in info) + "\n"
        head = ",".join(f"{h}" for h in heads) + "\n"

        with open(self.path, mode=self.mode) as f:
            f.write(info)
            f.write(head)

    def __call__(self, contents):
        content = ",".join(f"{c}" for c in contents) + "\n"
        with open(self.path, mode=self.mode) as f:
            f.write(content)
