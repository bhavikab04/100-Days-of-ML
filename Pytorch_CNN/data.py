import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loader(batch_size = 32):
    transform = transforms.Compose([
        transforms.ToTensor(),   # converts PIL image -> tensor, AND rescales pixel values 0-255 -> 0-1, AND converts HWC -> CHW
        # optionally: transforms.Normalize(mean, std) -- we'll add this after
    ])

    train_dataset = datasets.CIFAR10(root='/home/bhavika/Users/Bhavika/BITS - Acads/Computer_Vision/100-Days-of-ML/data', train=True, download=True, transform=transform)
    test_dataset  = datasets.CIFAR10(root='/home/bhavika/Users/Bhavika/BITS - Acads/Computer_Vision/100-Days-of-ML/data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    xb, yb = next(iter(train_loader))
    print(xb.shape)   # what do you expect? (batch_size, channels, H, W) --> (32, 3, 32, 32)
    return train_loader, test_loader