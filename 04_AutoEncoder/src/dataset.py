from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def load_data(path, batch_size=128):
    transform = transforms.ToTensor()

    train_dataset = datasets.CIFAR10(
        root=path,
        train=True,
        download=False,
        transform=transform,
    )

    test_dataset = datasets.CIFAR10(
        root=path,
        train=False,
        download=False,
        transform=transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader
