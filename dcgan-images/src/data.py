import torch
from torch.utils.data import DataLoader, TensorDataset


def loaders(n=256, batch_size=32, seed=42):
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(n, 1, 8, 8)
    for i in range(n):
        r = 1 + (i % 3)
        x[i, 0, 4 - r : 5 + r, 4 - r : 5 + r] = 1.0
        x[i] += 0.05 * torch.rand(1, 8, 8, generator=g)
    x = x.clamp(0, 1)
    return DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=True)
