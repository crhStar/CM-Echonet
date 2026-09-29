"""CM-EchoNet inference model.

The input is a dialogue tensor of cached WavLM features with shape [T, 768].
"""
import torch
from torch import nn


class WavLMEcho(nn.Module):
    def __init__(self, classes=4, hidden=128):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(768, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.state = nn.GRUCell(hidden, hidden)
        self.rate = nn.Sequential(nn.Linear(hidden, 32), nn.GELU(), nn.Linear(32, 3), nn.Softplus())
        self.write = nn.Linear(hidden, hidden)
        self.echo_proj = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.gate = nn.Sequential(nn.Linear(hidden * 2, 32), nn.GELU(), nn.Linear(32, 3))
        self.echo_drop = nn.Dropout(0.15)
        self.echo_alpha = nn.Parameter(torch.tensor(-2.0))
        self.head = nn.Linear(hidden * 2, classes)
        self.shift = nn.Linear(hidden * 2, 2)

    def forward(self, x):
        z = self.proj(x)
        g = z.new_zeros(1, z.size(-1))
        echo = z.new_zeros(1, 3, z.size(-1))
        logits, shifts = [], []
        for i in range(z.size(0)):
            zi = z[i:i + 1]
            g = self.state(zi, g)
            rates = self.rate(g)
            echo = torch.exp(-rates.unsqueeze(-1)) * echo + self.write(zi).unsqueeze(1)
            gate = torch.softmax(self.gate(torch.cat([g, echo.mean(1)], -1)), dim=-1).unsqueeze(-1)
            c = (self.echo_drop(echo) * gate).sum(1)
            c = self.echo_proj(c)
            q = torch.sigmoid(self.gate[0](torch.cat([g, c], -1)).mean(-1, keepdim=True))
            fused = torch.cat([g, zi + torch.sigmoid(self.echo_alpha) * q * c], -1)
            logits.append(self.head(fused))
            shifts.append(self.shift(fused))
        return torch.cat(logits), torch.cat(shifts)
