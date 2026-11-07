import torch
from torch.utils.data import DataLoader, TensorDataset

VOCAB = 20
LENGTH = 12


def loaders(n=400, batch_size=32, seed=42):
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 4, (n,), generator=g)
    x = torch.randint(4, VOCAB, (n, LENGTH), generator=g)
    for i, label in enumerate(y.tolist()):
        x[i, 0] = label
        x[i, 1] = label
        x[i, 2] = label
    n_train = int(0.8 * n)
    return (
        DataLoader(TensorDataset(x[:n_train], y[:n_train]), batch_size=batch_size, shuffle=True),
        DataLoader(TensorDataset(x[n_train:], y[n_train:]), batch_size=batch_size),
    )
