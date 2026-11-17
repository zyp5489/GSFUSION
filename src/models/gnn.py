import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (GATConv, GCNConv, GINConv,
                                global_mean_pool, global_max_pool,
                                MessagePassing)
from torch_geometric.utils import scatter

from .layers import GatedResidual, init_weights


class NodeToSuper(MessagePassing):
    """Phase 1: 节点 -> 超节点（门控消息）"""
    def __init__(self, channels):
        super().__init__(aggr='add')
        self.msg_net = nn.Sequential(
            nn.Linear(channels, channels), nn.GELU(),
            nn.Linear(channels, channels))
        self.gate_net = nn.Sequential(
            nn.Linear(channels * 2, channels // 2), nn.GELU(),
            nn.Linear(channels // 2, 1), nn.Sigmoid())

    def forward(self, x_node, x_super, edge_index):
        return self.propagate(edge_index, x_node=x_node, x_super=x_super,
                              size=(x_node.size(0), x_super.size(0)))

    def message(self, x_node_j, x_super_i):
        msg = self.msg_net(x_node_j)
        g = self.gate_net(torch.cat([x_node_j, x_super_i], dim=-1))
        return g * msg


class CrossGraphGNN(nn.Module):
    def __init__(self,
                 chem_in_channels, prot_in_channels,
                 hidden_channels=256, out_channels=256,
                 num_layers=3,
                 chem_gnn_type='gat', prot_gnn_type='gcn',
                 heads=4, dropout=0.1,
                 use_chem_graph=True,
                 use_prot_graph=True,
                 use_gcgim=True,
                 use_grc=True,
                 interaction_layers=None,
                 residual_mode='vector',
                 residual_init_bias=2.0,
                 using_fusion=True):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels    = out_channels
        self.num_layers      = num_layers
        self.using_fusion    = using_fusion

        self.use_chem_graph = use_chem_graph
        self.use_prot_graph = use_prot_graph
        self.use_grc        = use_grc
        self.use_gcgim      = use_gcgim and use_chem_graph and use_prot_graph

        if interaction_layers is None:
            interaction_layers = []
        if isinstance(interaction_layers, int):
            interaction_layers = [interaction_layers]
        self.interaction_layers = list(interaction_layers) if self.use_gcgim else []
        for layer in self.interaction_layers:
            if layer < 0 or layer >= num_layers:
                raise ValueError(f"Interaction layer index {layer} out of range")

        # ---- Chem 分支 ----
        if self.use_chem_graph:
            self.chem_in_proj  = nn.Linear(chem_in_channels, hidden_channels)
            self.chem_convs    = nn.ModuleList()
            self.chem_norms    = nn.ModuleList()
            self.chem_gated_res = nn.ModuleList() if self.use_grc else None
            for _ in range(num_layers):
                self.chem_convs.append(self._build_conv(chem_gnn_type, hidden_channels, heads, dropout))
                self.chem_norms.append(nn.LayerNorm(hidden_channels))
                if self.use_grc:
                    self.chem_gated_res.append(GatedResidual(
                        hidden_channels, mode=residual_mode, init_bias=residual_init_bias))
            self.chem_out_proj = nn.Linear(hidden_channels * 2, out_channels)
            self.chem_in_proj.apply(init_weights)
            self.chem_out_proj.apply(init_weights)

        # ---- Prot 分支 ----
        if self.use_prot_graph:
            self.prot_in_proj  = nn.Linear(prot_in_channels, hidden_channels)
            self.prot_convs    = nn.ModuleList()
            self.prot_norms    = nn.ModuleList()
            self.prot_gated_res = nn.ModuleList() if self.use_grc else None
            for _ in range(num_layers):
                self.prot_convs.append(self._build_conv(prot_gnn_type, hidden_channels, heads, dropout))
                self.prot_norms.append(nn.LayerNorm(hidden_channels))
                if self.use_grc:
                    self.prot_gated_res.append(GatedResidual(
                        hidden_channels, mode=residual_mode, init_bias=residual_init_bias))
            self.prot_out_proj = nn.Linear(hidden_channels * 2, out_channels)
            self.prot_in_proj.apply(init_weights)
            self.prot_out_proj.apply(init_weights)

        # ---- GCGIM 门控 ----
        if self.use_gcgim:
            self.gate_chems = nn.ModuleList()
            self.gate_prots = nn.ModuleList()
            for _ in range(num_layers):
                self.gate_chems.append(nn.Sequential(
                    nn.Linear(hidden_channels * 2, hidden_channels // 2),
                    nn.GELU(), nn.Linear(hidden_channels // 2, 1), nn.Sigmoid()))
                self.gate_prots.append(nn.Sequential(
                    nn.Linear(hidden_channels * 2, hidden_channels // 2),
                    nn.GELU(), nn.Linear(hidden_channels // 2, 1), nn.Sigmoid()))

        self.dropout = nn.Dropout(dropout)
        self.mepool  = global_mean_pool
        self.mapool  = global_max_pool

    @staticmethod
    def _build_conv(gnn_type, hidden_channels, heads, dropout):
        gnn_type = gnn_type.lower()
        if gnn_type == 'gat':
            return GATConv(hidden_channels, hidden_channels // heads,
                           heads=heads, concat=True, dropout=dropout)
        if gnn_type == 'gcn':
            return GCNConv(hidden_channels, hidden_channels)
        if gnn_type == 'gin':
            mlp = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.GELU(),
                                nn.Linear(hidden_channels, hidden_channels))
            return GINConv(mlp, train_eps=True)
        raise ValueError(f"Unknown gnn_type: {gnn_type}")

    def forward(self, chem_data, prot_data, chem_sample_ids=None,
                prot_sample_ids=None, num_samples=None):
        chem_emb_out, prot_emb_out = None, None
        if not self.use_chem_graph and not self.use_prot_graph:
            return chem_emb_out, prot_emb_out

        if self.use_gcgim:
            if chem_sample_ids is None or prot_sample_ids is None:
                raise ValueError("开启 GCGIM 时必须提供 sample_ids")
            chem_sample_ids = chem_sample_ids.long()
            prot_sample_ids = prot_sample_ids.long()
            B = int(num_samples) if num_samples is not None else int(chem_sample_ids.size(0))

        if self.use_chem_graph:
            chem_x      = self.chem_in_proj(chem_data.x)
            chem_edge   = chem_data.edge_index
            chem_batch  = getattr(chem_data, 'batch', None)
            if chem_batch is None:
                chem_batch = chem_x.new_zeros(chem_x.size(0), dtype=torch.long)
            B_c = int(chem_batch.max().item()) + 1
            chem_layer_pools = []

        if self.use_prot_graph:
            prot_x      = self.prot_in_proj(prot_data.x)
            prot_edge   = prot_data.edge_index
            prot_batch  = getattr(prot_data, 'batch', None)
            if prot_batch is None:
                prot_batch = prot_x.new_zeros(prot_x.size(0), dtype=torch.long)
            B_p = int(prot_batch.max().item()) + 1
            prot_layer_pools = []

        for layer_idx in range(self.num_layers):
            if self.use_chem_graph:
                chem_x_new = self.chem_convs[layer_idx](chem_x, chem_edge)
                chem_x = self.chem_gated_res[layer_idx](chem_x_new, chem_x) if self.use_grc else chem_x_new
                chem_x = F.relu(self.chem_norms[layer_idx](chem_x))
                chem_x = self.dropout(chem_x)

            if self.use_prot_graph:
                prot_x_new = self.prot_convs[layer_idx](prot_x, prot_edge)
                prot_x = self.prot_gated_res[layer_idx](prot_x_new, prot_x) if self.use_grc else prot_x_new
                prot_x = F.relu(self.prot_norms[layer_idx](prot_x))
                prot_x = self.dropout(prot_x)

            if self.use_gcgim and (layer_idx in self.interaction_layers):
                chem_global_g = self.mepool(chem_x, chem_batch) + self.mapool(chem_x, chem_batch)
                prot_global_g = self.mepool(prot_x, prot_batch) + self.mapool(prot_x, prot_batch)

                chem_global_s = chem_global_g[chem_sample_ids]
                prot_global_s = prot_global_g[prot_sample_ids]

                prot_msg_g = scatter(prot_global_s, chem_sample_ids, dim=0, dim_size=B_c, reduce='mean')
                chem_msg_g = scatter(chem_global_s, prot_sample_ids, dim=0, dim_size=B_p, reduce='mean')

                prot_msg = prot_msg_g[chem_batch]
                chem_msg = chem_msg_g[prot_batch]

                gate_chem = self.gate_chems[layer_idx](torch.cat([chem_x, prot_msg], dim=-1))
                gate_prot = self.gate_prots[layer_idx](torch.cat([prot_x, chem_msg], dim=-1))

                chem_x = chem_x + gate_chem * prot_msg
                prot_x = prot_x + gate_prot * chem_msg

            if self.use_chem_graph:
                chem_layer_pools.append(torch.cat([
                    self.mapool(chem_x, chem_batch),
                    self.mepool(chem_x, chem_batch)], dim=-1))
            if self.use_prot_graph:
                prot_layer_pools.append(torch.cat([
                    self.mapool(prot_x, prot_batch),
                    self.mepool(prot_x, prot_batch)], dim=-1))

        if self.use_chem_graph:
            chem_emb_out = self.chem_out_proj(chem_layer_pools[-1])
        if self.use_prot_graph:
            prot_emb_out = self.prot_out_proj(prot_layer_pools[-1])

        return chem_emb_out, prot_emb_out