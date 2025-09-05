import torch
from torch.utils.data import Dataset
from .idxio import load_idx
import os
from typing import Dict, List
from unifiedfl.utils.logging import get_logger

logger = get_logger("ugnn.data")

class MNISTLike(Dataset):
    def __init__(self, root_dir, split: str, val_size=0.2, transform=None, seed=42):
        self.root_dir = root_dir
        prefix = "train" if split in ["train", "val"] else "t10k"
        images_filename = prefix + "-images-idx3-ubyte.gz"
        labels_filename = prefix + "-labels-idx1-ubyte.gz"
        pert_filename   = prefix + "-pert-idx1-ubyte.gz"

        self.images = torch.from_numpy(load_idx(os.path.join(self.root_dir, images_filename)))
        self.labels = torch.from_numpy(load_idx(os.path.join(self.root_dir, labels_filename)))
        self.perts  = torch.from_numpy(load_idx(os.path.join(self.root_dir, pert_filename)))
        assert len(self.images) == len(self.labels) == len(self.perts)

        num_samples = self.images.size(0)
        n_val = int(val_size * num_samples)
        n_train = num_samples - n_val

        g = torch.Generator().manual_seed(seed)
        indices = torch.randperm(num_samples, generator=g)
        train_indices = indices[:n_train]
        val_indices   = indices[n_train:]

        if split == "train":
            idx = train_indices
        elif split == "val":
            idx = val_indices
        else:
            idx = torch.arange(num_samples)

        self.images = self.images[idx]
        self.labels = self.labels[idx]
        self.perts  = self.perts[idx]
        self.transform = transform

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].float().unsqueeze(0) / 255.0
        label = self.labels[idx]
        pert  = self.perts[idx]
        if self.transform: img = self.transform(img)
        return img, label, pert

class MorphoMNISTDistShiftDataset(Dataset):
    def __init__(self, dataset: MNISTLike, perturbation_to_model: Dict[int, int]):
        self.dataset = dataset
        self.perturbation_to_model = perturbation_to_model
        self.model_to_indices = {}
        for idx, pert in enumerate(self.dataset.perts):
            p = int(pert.item())
            if p in self.perturbation_to_model:
                m = self.perturbation_to_model[p]
                self.model_to_indices.setdefault(m, []).append(idx)
        for m, ind in self.model_to_indices.items():
            logger.info(f"Model {m} -> {len(ind)} samples")
        self.length = min(len(v) for v in self.model_to_indices.values())

    def __len__(self): return self.length

    def __getitem__(self, i):
        batch = {}
        for m, inds in self.model_to_indices.items():
            j = inds[i % len(inds)]
            img, lab, _ = self.dataset[j]
            batch[m] = (img, lab)
        return batch

def multi_mlp_collate_fn(batch):
    collated = {}
    mlp_indices = batch[0].keys()
    for m in mlp_indices:
        imgs, labels = [], []
        for item in batch:
            im, la = item[m]
            imgs.append(im); labels.append(la)
        collated[m] = (torch.stack(imgs), torch.tensor(labels))
    return collated

class PerturbationDataset(Dataset):
    def __init__(self, dataset: MNISTLike, perturbation_levels: List[int]):
        self.dataset = dataset
        self.indices = [i for i, p in enumerate(self.dataset.perts) if int(p.item()) in perturbation_levels]
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        j = self.indices[idx]
        img, lab, pert = self.dataset[j]
        return img, lab, pert

class Pert0Dataset(PerturbationDataset):
    def __init__(self, dataset): super().__init__(dataset, [0])
class Pert1Dataset(PerturbationDataset):
    def __init__(self, dataset): super().__init__(dataset, [1])
class Pert2Dataset(PerturbationDataset):
    def __init__(self, dataset): super().__init__(dataset, [2])

class MixedPerturbationDataset(Dataset):
    def __init__(self, dataset: MNISTLike, seed=42):
        self.dataset = dataset
        g = torch.Generator().manual_seed(seed)
        self.indices = torch.randperm(len(self.dataset), generator=g).tolist()
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        j = self.indices[idx]
        x, y, _ = self.dataset[j]
        return x, y

class MedMNISTDistShift(Dataset):
    def __init__(self, dataset, cluster_to_indices: dict):
        self.dataset = dataset
        self.model_to_indices = cluster_to_indices
        for m, ind in self.model_to_indices.items():
            logger.info(f"Cluster {m} -> {len(ind)} samples")
        self.length = min(len(v) for v in self.model_to_indices.values())

    def __len__(self): return self.length

    def __getitem__(self, idx):
        batch = {}
        for m, inds in self.model_to_indices.items():
            j = int(inds[idx % len(inds)])
            img, lab = self.dataset[j]
            batch[m] = (img, lab)
        return batch
