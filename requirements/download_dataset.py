import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def get_mnist_loaders(
    root: str = "./data",
    batch_size: int = 64,
    val_frac: float = 0.1,
    device: torch.device | None = None,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create MNIST train/val/test DataLoaders ready for use in a network.

    :param root: directory to download/store the MNIST data.
    :param batch_size: batch size for loaders.
    :param val_frac: fraction of the training set to hold out for validation (0..1).
    :param device: torch.device used to decide pin_memory (if None, auto-detect).
    :param num_workers: DataLoader num_workers (set to 0 on some platforms if you hit issues).
    :returns: (train_loader, val_loader, test_loader)
    """
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    # Standard MNIST transforms: PIL -> tensor in [0,1] and normalized by MNIST mean/std
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # -> shape [1, 28, 28], values in [0,1]
            transforms.Normalize((0.1307,), (0.3081,)),  # commonly used mean/std for MNIST
        ]
    )

    # Download datasets (train includes the classic 60k examples)
    train_full = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=transform)

    # Split training into train + val
    n_total = len(train_full)  # 60000
    n_val = int(n_total * val_frac)
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(42)  # reproducible split
    train_dataset, val_dataset = random_split(train_full, [n_train, n_val], generator=generator)

    # pin_memory is useful only for CUDA; MPS/CPU should not use it (avoids the warning you saw)
    pin_memory = True if device.type == "cuda" else False

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader
