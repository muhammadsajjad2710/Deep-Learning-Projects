import torch
from sklearn.datasets import load_digits
from torch.utils.data import DataLoader, TensorDataset


def loaders(batch_size=32, noise=0.4, seed=42):
    digits = load_digits()
    clean = torch.tensor(digits.data, dtype=torch.float32).reshape(-1, 1, 8, 8) / 16.0
    g = torch.Generator().manual_seed(seed)
    noisy = (clean + noise * torch.rand_like(clean, generator=g)).clamp(0, 1)
    n = int(0.8 * len(clean))
    perm = torch.randperm(len(clean), generator=g)
    train = TensorDataset(noisy[perm[:n]], clean[perm[:n]])
    test = TensorDataset(noisy[perm[n:]], clean[perm[n:]])
    return DataLoader(train, batch_size=batch_size, shuffle=True), DataLoader(test, batch_size=batch_size)
