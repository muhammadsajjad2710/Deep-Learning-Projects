import torch
from torch.utils.data import DataLoader, TensorDataset


def loaders(n=500, length=16, batch_size=32, seed=42):
    t = torch.linspace(0, 20, n + length)
    series = torch.sin(t) + 0.1 * torch.sin(3 * t)
    x, y = [], []
    for i in range(n):
        x.append(series[i : i + length])
        y.append(series[i + length])
    x = torch.stack(x).unsqueeze(-1)
    y = torch.stack(y)
    n_train = int(0.8 * n)
    return (
        DataLoader(TensorDataset(x[:n_train], y[:n_train]), batch_size=batch_size, shuffle=True),
        DataLoader(TensorDataset(x[n_train:], y[n_train:]), batch_size=batch_size),
    )
