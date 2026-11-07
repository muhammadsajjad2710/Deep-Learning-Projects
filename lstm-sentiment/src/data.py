import torch
from torch.utils.data import DataLoader, TensorDataset

POS = {1, 2, 3}
NEG = {7, 8, 9}
VOCAB = 12
LENGTH = 8


def loaders(n=400, batch_size=32, seed=42):
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 2, (n,), generator=g)
    x = torch.randint(4, 7, (n, LENGTH), generator=g)
    for i, label in enumerate(y.tolist()):
        words = list(POS if label == 1 else NEG)
        x[i, 0] = words[i % len(words)]
        x[i, 1] = words[(i + 1) % len(words)]
    n_train = int(0.8 * n)
    return (
        DataLoader(TensorDataset(x[:n_train], y[:n_train]), batch_size=batch_size, shuffle=True),
        DataLoader(TensorDataset(x[n_train:], y[n_train:]), batch_size=batch_size),
    )
