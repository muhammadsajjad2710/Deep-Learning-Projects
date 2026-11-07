import torch
from torch import nn
from src.data import LENGTH, VOCAB, loaders


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, 32)
        self.pos = nn.Parameter(torch.randn(1, LENGTH, 32) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(32, 4)

    def forward(self, x):
        h = self.embed(x) + self.pos
        h = self.encoder(h)
        return self.head(h.mean(1))


def train(epochs=6, seed=42):
    torch.manual_seed(seed)
    train_loader, test_loader = loaders()
    model = TinyTransformer()
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
