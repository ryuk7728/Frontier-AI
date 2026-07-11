"""Reproducible CIFAR-10 data loading."""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def load_data(
    path: str,
    batch_size: int = 128,
    validation_size: int = 5000,
    workers: int = 4,
    seed: int = 42,
    download: bool = False,
):
    evaluation_transform = transforms.ToTensor()
    # A horizontal flip preserves the reconstruction task and reduces memorization.
    training_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    augmented_train = datasets.CIFAR10(path, train=True, download=download, transform=training_transform)
    plain_train = datasets.CIFAR10(path, train=True, download=False, transform=evaluation_transform)
    test_dataset = datasets.CIFAR10(path, train=False, download=False, transform=evaluation_transform)

    indices = torch.randperm(len(augmented_train), generator=torch.Generator().manual_seed(seed)).tolist()
    train_indices = indices[:-validation_size]
    validation_indices = indices[-validation_size:]
    train_dataset = Subset(augmented_train, train_indices)
    validation_dataset = Subset(plain_train, validation_indices)
    common = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        **common,
    )
    validation_loader = DataLoader(validation_dataset, shuffle=False, **common)
    test_loader = DataLoader(test_dataset, shuffle=False, **common)
    return train_loader, validation_loader, test_loader
