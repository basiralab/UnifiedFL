import torch
from torch.utils.data import DataLoader
from medmnist.dataset import PathMNIST
from torchvision import transforms

def build_medmnist_splits(batch_size=1024, seed=42):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    tt = (lambda x: torch.tensor(x[0], dtype=torch.long))
    tr = PathMNIST(split="train", download=True, transform=tfm, target_transform=tt)
    va = PathMNIST(split="val",   download=True, transform=tfm, target_transform=tt)
    te = PathMNIST(split="test",  download=True, transform=tfm, target_transform=tt)
    g = torch.Generator().manual_seed(seed)
    return (
        DataLoader(tr, batch_size=batch_size, shuffle=True,  generator=g),
        DataLoader(va, batch_size=batch_size, shuffle=False, generator=g),
        DataLoader(te, batch_size=batch_size, shuffle=False, generator=g),
    ), (tr, va, te)

def build_distshift_loader(distshift_dataset, batch_size=1024, seed=42):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(distshift_dataset, batch_size=batch_size, shuffle=True, generator=g)
