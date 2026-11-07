import torch


def content_and_style(size=16, seed=42):
    g = torch.Generator().manual_seed(seed)
    content = torch.zeros(1, 3, size, size)
    content[:, 0, 4:12, 4:12] = 1.0
    style = torch.zeros(1, 3, size, size)
    style[:, 1] = torch.linspace(0, 1, size).view(1, 1, size)
    style[:, 2] = torch.rand(1, size, size, generator=g)
    return content, style
