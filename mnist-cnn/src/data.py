import torch
from sklearn.datasets import load_digits
from torch.utils.data import DataLoader, TensorDataset


def loaders(batch_size=32, seed=42):
    digits = load_digits()
    x = torch.tensor(digits.data, dtype=torch.float32).reshape(-1, 1, 8, 8) / 16.0
    y = torch.tensor(digits.target, dtype=torch.long)
    g = torch.Generator().manual_seed(seed)
    n = int(0.8 * len(x))
    perm = torch.randperm(len(x), generator=g)
    train_idx, test_idx = perm[:n], perm[n:]
    train_ds = TensorDataset(x[train_idx], y[train_idx])
    test_ds = TensorDataset(x[test_idx], y[test_idx])
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(test_ds, batch_size=batch_size)
