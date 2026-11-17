import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerProjectionShared(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, out_dim=128, num_layers=3):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
            )
            for _ in range(num_layers)
        ])

    def forward(self, x):
        outs = [self.proj[l](x[:, l, :]) for l in range(x.size(1))]
        return torch.stack(outs, dim=1)


def cross_view_contrastive(z1, z2, proj1, proj2, t=0.1):
    with torch.autocast(device_type='cuda', enabled=False):
        z1 = proj1(z1).float()
        z2 = proj2(z2).float()

        B = z1.size(0)
        logits = z1 @ z2.t() / t
        labels = torch.arange(B, device=z1.device)

        loss_12 = F.cross_entropy(logits, labels)
        loss_21 = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_12 + loss_21)


def layerwise_view_contrastive(chem_emb_all, chem_fused2,
                               proj1, proj2, t=0.1,
                               layers_use=None, decay=0.15):
    return cross_view_contrastive(
        chem_emb_all[:, 0, :], chem_fused2,
        proj1, proj2, t=t,
    )


class CrossViewProjection(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            return F.normalize(self.net(x), dim=-1)