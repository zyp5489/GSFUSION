import torch
import torch.nn as nn


class DynamicGatedFusion(nn.Module):
    def __init__(self, input_dim_per_branch, num_branches):
        super().__init__()
        self.num_branches = num_branches
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim_per_branch * 2, input_dim_per_branch // 2),
            nn.GELU(),
            nn.Linear(input_dim_per_branch // 2, num_branches))

    def forward(self, *branch_list):
        B = branch_list[0].shape[0]
        aggregate   = torch.cat(branch_list[0:2], dim=-1)
        gate_logits = self.gate_net(aggregate)
        gates       = torch.softmax(gate_logits, dim=-1)

        weighted = [gates[:, i:i + 1] * feat for i, feat in enumerate(branch_list)]
        return torch.cat(weighted, dim=-1)