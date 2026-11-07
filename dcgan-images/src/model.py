import torch
from torch import nn
from src.data import loaders


class Generator(nn.Module):
    def __init__(self, z_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z).view(-1, 1, 8, 8)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def train(epochs=4, seed=42, z_dim=16):
    torch.manual_seed(seed)
    loader = loaders()
    gen = Generator(z_dim)
    disc = Discriminator()
    g_opt = torch.optim.Adam(gen.parameters(), lr=2e-3)
    d_opt = torch.optim.Adam(disc.parameters(), lr=2e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    last_g = last_d = 0.0
    for _ in range(epochs):
        for (real,) in loader:
            batch = real.size(0)
            real_y = torch.ones(batch, 1)
            fake_y = torch.zeros(batch, 1)
            z = torch.randn(batch, z_dim)
            fake = gen(z).detach()
            d_opt.zero_grad()
            d_loss = loss_fn(disc(real), real_y) + loss_fn(disc(fake), fake_y)
            d_loss.backward()
            d_opt.step()
            z = torch.randn(batch, z_dim)
            g_opt.zero_grad()
            g_loss = loss_fn(disc(gen(z)), real_y)
            g_loss.backward()
            g_opt.step()
            last_g, last_d = float(g_loss.detach()), float(d_loss.detach())
    return {"generator": gen, "discriminator": disc}, {"g_loss": last_g, "d_loss": last_d}
