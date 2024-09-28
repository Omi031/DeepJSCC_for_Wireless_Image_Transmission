import numpy as np
import matplotlib.pyplot as plt


def show_and_save(file_name, img, lib):
    if lib == "pytorch":
        npimg = np.transpose(img.numpy(), (1, 2, 0))
    elif lib == "tensorflow":
        pass
    # fig = plt.figure(dpi=200)
    # fig.suptitle(file_name, fontsize=14, fontweight="bold")
    plt.imshow(npimg)
    plt.imsave(file_name, npimg)


def plot_loss(loss_list):
    plt.figure(figsize=(10, 5))
    plt.title("Loss During Training")
    plt.plot(loss_list, label="Loss")

    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
