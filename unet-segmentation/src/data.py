import torch
from torch.utils.data import DataLoader, TensorDataset


def loaders(n=200, size=16, batch_size=16, seed=42):
    g = torch.Generator().manual_seed(seed)
    images = torch.rand(n, 1, size, size, generator=g) * 0.15
    masks = torch.zeros(n, 1, size, size)
    for i in range(n):
        cy = int(torch.randint(4, 12, (1,), generator=g))
        cx = int(torch.randint(4, 12, (1,), generator=g))
        r = int(torch.randint(2, 4, (1,), generator=g))
        yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
        blob = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r
        masks[i, 0] = blob.float()
        images[i, 0] = torch.where(blob, torch.tensor(0.9), images[i, 0])
    n_train = int(0.8 * n)
    return (
        DataLoader(TensorDataset(images[:n_train], masks[:n_train]), batch_size=batch_size, shuffle=True),
        DataLoader(TensorDataset(images[n_train:], masks[n_train:]), batch_size=batch_size),
    )
