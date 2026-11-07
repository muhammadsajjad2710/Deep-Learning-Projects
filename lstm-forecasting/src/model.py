import torch
from torch import nn
from src.data import loaders


class ForecastLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 32, batch_first=True)
        self.head = nn.Linear(32, 1)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.head(hidden[-1]).squeeze(-1)


def train(epochs=8, seed=42):
    torch.manual_seed(seed)
    train_loader, test_loader = loaders()
    model = ForecastLSTM()
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
            count += len(yb)
    return model, {"mse": mse / count}
