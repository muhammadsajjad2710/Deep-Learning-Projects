import torch
from torch.utils.data import DataLoader, TensorDataset


def loaders(n=800, batch_size=32, seed=42):
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 10, (n,), generator=g)
    x = torch.rand(n, 3, 16, 16, generator=g) * 0.15
    for i, label in enumerate(y):
        c = int(label) % 3
        x[i, c] += 0.7
        band = int(label) % 8
        x[i, :, band : band + 2, :] += 0.4
    x = x.clamp(0, 1)
    n_train = int(0.8 * n)
    train_ds = TensorDataset(x[:n_train], y[:n_train])
    test_ds = TensorDataset(x[n_train:], y[n_train:])
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(test_ds, batch_size=batch_size)
