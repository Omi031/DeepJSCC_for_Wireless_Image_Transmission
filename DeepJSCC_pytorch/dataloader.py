import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def dataloader(batch_size=64, train=True, root="data", subset=False):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(
        root=root, train=train, transform=transform, download=True
    )
    if subset == True:
        subset_idx = list(range(64 * 3))
        dataset = Subset(dataset, subset_idx)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader
