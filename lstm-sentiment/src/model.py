import torch
from torch import nn
from src.data import VOCAB, loaders


class SentimentLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, 16)
        self.lstm = nn.LSTM(16, 32, batch_first=True)
        self.head = nn.Linear(32, 2)

    def forward(self, x):
        _, (hidden, _) = self.lstm(self.embed(x))
        return self.head(hidden[-1])


def train(epochs=6, seed=42):
    torch.manual_seed(seed)
    train_loader, test_loader = loaders()
    model = SentimentLSTM()
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
