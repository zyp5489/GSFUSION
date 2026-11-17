import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)


def masked_mean(x, mask, dim=1, eps=1e-8):
    if mask.dim() == x.dim() - 1:
        mask = mask.unsqueeze(-1)
    mask = mask.float()
    return (x * mask).sum(dim=dim) / mask.sum(dim=dim).clamp(min=eps)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class GatedResidual(nn.Module):
    """门控残差：g * h_new + (1-g) * h_old"""
    def __init__(self, hidden_dim, mode='vector', init_bias=2.0):
        super().__init__()
        self.mode = mode
        out_dim = hidden_dim if mode == 'vector' else 1
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        last = self.gate[-1]
        nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias, init_bias)

    def forward(self, h_new, h_old):
        g = torch.sigmoid(self.gate(torch.cat([h_new, h_old], dim=-1)))
        return g * h_new + (1.0 - g) * h_old