import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def dataloader(batch_size=64, train=True, subset=False):
    dataroot = "data"
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(
        root=dataroot, train=train, transform=transform, download=True
    )
    subset_idx = list(range(10))
    if subset == True:
        dataset = Subset(dataset, subset_idx)
    if train == True:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader
