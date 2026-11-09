"""EuroSAT loader (first dataset). ISIC 2019 and Cassava loaders will follow the same interface."""
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def _tf(train: bool, img_size: int):
    ops = [transforms.Resize((img_size, img_size))]
    if train:
        ops += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
    ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(ops)


def get_eurosat_loaders(root="data", img_size=224, batch_size=64, seed=0, num_workers=4):
    """70/15/15 random split fixed by `seed`, so every model sees exactly the same
    data when the seed is the same."""
    train_ds = datasets.EuroSAT(root, download=True, transform=_tf(True, img_size))
    eval_ds = datasets.EuroSAT(root, download=False, transform=_tf(False, img_size))
    n = len(train_ds)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_tr, n_va = int(0.70 * n), int(0.15 * n)
    idx_tr, idx_va, idx_te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]

    mk = lambda ds, idx, sh: DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=sh,
                                        num_workers=num_workers, pin_memory=True)
    return mk(train_ds, idx_tr, True), mk(eval_ds, idx_va, False), mk(eval_ds, idx_te, False), len(train_ds.classes)
