import torch
from torch import nn
from src.data import loaders


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU())
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(nn.Conv2d(8, 16, 3, padding=1), nn.ReLU())
        self.up = nn.ConvTranspose2d(16, 8, 2, stride=2)
        self.dec = nn.Sequential(nn.Conv2d(16, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 1, 1))

    def forward(self, x):
        skip = self.enc(x)
        up = self.up(self.bottleneck(self.pool(skip)))
        return self.dec(torch.cat([up, skip], dim=1))


def dice(pred, target, eps=1e-6):
    pred = (torch.sigmoid(pred) > 0.5).float()
    inter = (pred * target).sum()
    return float((2 * inter + eps) / (pred.sum() + target.sum() + eps))


def train(epochs=12, seed=42):
    torch.manual_seed(seed)
    train_loader, test_loader = loaders()
    model = TinyUNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    model.eval()
    scores = []
    with torch.no_grad():
        for xb, yb in test_loader:
            scores.append(dice(model(xb), yb))
    return model, {"dice": sum(scores) / len(scores)}
