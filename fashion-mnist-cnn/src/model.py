import torch
from torch import nn
from src.data import loaders


class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 10),
        )

    def forward(self, x):
        return self.net(x)


def train(epochs=5, seed=42):
    torch.manual_seed(seed)
    train_loader, test_loader = loaders()
    model = FashionCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            correct += int((model(xb).argmax(1) == yb).sum())
            total += len(yb)
    return model, {"accuracy": correct / total}
