import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )
        self.mu = nn.Linear(512, 128)
        self.logvar = nn.Linear(512, 128)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x, stochastic):
        shared = self.shared(x)
        mu = self.mu(shared)
        logvar = self.logvar(shared)
        if not stochastic:
            return mu
        z = self.reparametrize(mu, logvar)
        return z, mu, logvar

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        z, mu, logvar = self.encoder(x, stochastic=True)
        y = self.classifier(z)
        return z, y, mu, logvar