import torch
from torch.utils.data import DataLoader, TensorDataset

# Ten clothing-like grayscale patterns (shirt, trouser, pullover, dress, coat,
# sandal, shirt-stripe, sneaker, bag, boot) so the task matches Fashion-MNIST.
LABELS = 10


def loaders(n=900, batch_size=32, seed=42):
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, LABELS, (n,), generator=g)
    x = torch.zeros(n, 1, 16, 16)
    for i, label in enumerate(y.tolist()):
        noise = torch.rand(16, 16, generator=g) * 0.08
        img = noise.clone()
        if label in (0, 2, 4, 6):
            img[3:13, 4:12] = 0.8
            img[3:5, 4:12] = 0.95
        elif label == 1:
            img[2:14, 6:10] = 0.85
        elif label == 3:
            img[2:14, 5:11] = 0.75
            img[2:4, 4:12] = 0.9
        elif label in (5, 7, 9):
            img[10:15, 3:13] = 0.8
            img[8:11, 7:9] = 0.7
        else:
            img[4:12, 4:12] = 0.7
            img[6:10, 6:10] = 0.95
        img[:, label : label + 1] += 0.15
        x[i, 0] = img.clamp(0, 1)
    n_train = int(0.8 * n)
    return (
        DataLoader(TensorDataset(x[:n_train], y[:n_train]), batch_size=batch_size, shuffle=True),
        DataLoader(TensorDataset(x[n_train:], y[n_train:]), batch_size=batch_size),
    )
