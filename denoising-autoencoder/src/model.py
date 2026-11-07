import torch
from torch import nn
from src.data import loaders


class DenoiseAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 4, 3, padding=1), nn.ReLU())
        self.decoder = nn.Sequential(nn.Conv2d(4, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 1, 3, padding=1), nn.Sigmoid())

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train(epochs=5, seed=42):
    torch.manual_seed(seed)
    train_loader, test_loader = loaders()
    model = DenoiseAE()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    model.eval()
    mse = 0.0
    count = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            mse += float(loss_fn(model(xb), yb)) * len(xb)
            count += len(xb)
    return model, {"mse": mse / count}
