import torch
from torch import nn
from src.data import content_and_style


class FeatureNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h1 = torch.relu(self.conv1(x))
        h2 = torch.relu(self.conv2(h1))
        return h1, h2


def gram(feat):
    b, c, h, w = feat.shape
    f = feat.view(b, c, h * w)
    return torch.bmm(f, f.transpose(1, 2)) / (c * h * w)


def train(steps=40, seed=42):
    torch.manual_seed(seed)
    content, style = content_and_style()
    net = FeatureNet()
    image = content.clone().requires_grad_(True)
    opt = torch.optim.Adam([image], lr=5e-2)
    c1, c2 = net(content)
    _, s2 = net(style)
    start = None
    last = None
    for _ in range(steps):
        opt.zero_grad()
        h1, h2 = net(image)
        loss = torch.mean((h1 - c1) ** 2) + 1e3 * torch.mean((gram(h2) - gram(s2)) ** 2)
        if start is None:
            start = float(loss.detach())
        loss.backward()
        opt.step()
        last = float(loss.detach())
        image.data.clamp_(0, 1)
    return {"image": image.detach(), "features": net}, {"start_loss": start, "final_loss": last}
