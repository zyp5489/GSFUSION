import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

from .layers       import LayerNorm, masked_mean, init_weights
from .gnn          import CrossGraphGNN
from .cross_attention import cross_AFT
from .contrastive  import (cross_view_contrastive,
                           layerwise_view_contrastive,
                           CrossViewProjection)
from .fusion       import DynamicGatedFusion


class DTAProtGraphChemGraph3(nn.Module):
    def __init__(self, **param_dict):
        super().__init__()

        # ---- 消融开关 ----
        self.use_prot_seq = param_dict.get('use_prot_seq', True)
        self.use_comp_seq = param_dict.get('use_comp_seq', True)
        self.use_prot_graph = param_dict.get('use_prot_graph', True)
        self.use_comp_graph = param_dict.get('use_comp_graph', True)

        self.use_cross_attention = param_dict.get('use_cross_attention', True)
        self.use_gcgim = param_dict.get('use_gcgim', True)
        self.use_grc   = param_dict.get('use_grc', True)
        self.use_cvcl  = param_dict.get('use_cvcl', True)

        if not (self.use_prot_seq and self.use_comp_seq):
            self.use_cross_attention = False

        self.chem_initial_norm = LayerNorm(param_dict["chem_in_features"])
        self.prot_initial_norm = LayerNorm(param_dict["prot_in_features"])

        num_layers = 5
        interaction_layers = [1] if (self.use_gcgim and self.use_prot_graph and self.use_comp_graph) else []

        self.cross_gnn = CrossGraphGNN(
            chem_in_channels=param_dict["chem_in_features"],
            prot_in_channels=param_dict["prot_in_features"],
            hidden_channels=256, out_channels=256, num_layers=num_layers,
            chem_gnn_type='gat', prot_gnn_type='gcn',
            heads=4, dropout=0.1,
            use_chem_graph=self.use_comp_graph,
            use_prot_graph=self.use_prot_graph,
            use_gcgim=self.use_gcgim,
            use_grc=self.use_grc,
            interaction_layers=interaction_layers,
            residual_mode='vector', residual_init_bias=2.0, using_fusion=True)

        self.fp_linear = nn.Sequential(
            nn.Linear(2048, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1))

        self.using_lstm = param_dict.get("using_lstm", True)
        if self.using_lstm:
            self.chem_seq_lstm = nn.LSTM(input_size=256, hidden_size=128,
                                         num_layers=2, batch_first=True,
                                         bidirectional=True, dropout=0.0)
            self.prot_seq_lstm = nn.LSTM(input_size=256, hidden_size=128,
                                         num_layers=2, batch_first=True,
                                         bidirectional=True, dropout=0.0)

        self.chem_projs = nn.Sequential(
            nn.Linear(768, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1))
        self.prot_projs = nn.Sequential(
            nn.Linear(2560, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1))

        if self.use_cross_attention:
            self.cross_att = cross_AFT(param_dict, 256, 256)

        # ---- 动态分支 ----
        self.branch_names = []
        if self.use_comp_graph:  self.branch_names.append('comp_graph')
        if self.use_prot_graph:  self.branch_names.append('prot_graph')
        self.branch_names.append('fp')
        if self.use_comp_seq:    self.branch_names.append('comp_seq')
        if self.use_prot_seq:    self.branch_names.append('prot_seq')
        if self.use_cross_attention:
            self.branch_names.append('cross_comp')
            self.branch_names.append('cross_prot')

        num_branches = len(self.branch_names)
        print(f"Model initialized with {num_branches} active branches: {self.branch_names}")

        self.seq_expert = nn.Sequential(
            nn.Linear(256 * num_branches, 1024),
            nn.LayerNorm(1024), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

        self.dynamic_gate = DynamicGatedFusion(
            input_dim_per_branch=256, num_branches=num_branches)

        self.chem_graph_proj = CrossViewProjection(in_dim=256, out_dim=256)
        self.chem_seq_proj   = CrossViewProjection(in_dim=256, out_dim=256)
        self.prot_graph_proj = CrossViewProjection(in_dim=256, out_dim=256)
        self.prot_seq_proj   = CrossViewProjection(in_dim=256, out_dim=256)
        self.cross_graph_proj= CrossViewProjection(in_dim=256, out_dim=256)
        self.cross_seq_proj  = CrossViewProjection(in_dim=256, out_dim=256)

        self.seq_expert.apply(init_weights)
        self.fp_linear.apply(init_weights)
        self.chem_projs.apply(init_weights)
        self.prot_projs.apply(init_weights)
        for m in [self.chem_graph_proj, self.chem_seq_proj,
                  self.prot_graph_proj, self.prot_seq_proj,
                  self.cross_graph_proj, self.cross_seq_proj]:
            m.apply(init_weights)

    def forward(self, chem_fp, chem_graph, prot_graph, batch_ligand_token,
                atom_pad_mask, batch_drug_ids, batch_target_ids,
                full_affinity_train, return_aux_loss=False):

        idx_drug   = batch_drug_ids.long()
        idx_target = batch_target_ids.long()
        B = chem_fp.shape[0]

        chem_graph.x = self.chem_initial_norm(chem_graph.x)
        prot_graph.x = self.prot_initial_norm(prot_graph.x[:, 0:39])

        repr_fp = self.fp_linear(chem_fp)

        chem_graph_rep_diff_org = self.chem_projs(batch_ligand_token)
        prot_graph.x2 = self.prot_projs(prot_graph.x2)
        prot_node_dense, prot_node_mask = to_dense_batch(prot_graph.x2, prot_graph.batch)

        chem_emb, prot_emb = None, None
        chem_emb_all, prot_emb_all = None, None
        if self.use_comp_graph or self.use_prot_graph:
            chem_emb_all, prot_emb_all = self.cross_gnn(
                chem_graph, prot_graph,
                chem_sample_ids=idx_drug, prot_sample_ids=idx_target, num_samples=B)
            chem_emb = chem_emb_all[idx_drug] if self.use_comp_graph else None
            prot_emb = prot_emb_all[idx_target] if self.use_prot_graph else None

        chem_graph_rep_diff, prot_graph_rep_diff = None, None
        if self.using_lstm:
            with torch.autocast(device_type='cuda', enabled=False):
                if self.use_comp_seq:
                    chem_lstm_out, _ = self.chem_seq_lstm(chem_graph_rep_diff_org)
                    chem_graph_rep_diff = masked_mean(chem_lstm_out, mask=atom_pad_mask, dim=1)
                if self.use_prot_seq:
                    if prot_node_dense.shape[1] > 1024:
                        prot_node_dense = prot_node_dense[:, :1024, :]
                        prot_node_mask  = prot_node_mask[:, :1024]
                    prot_lstm_out, _ = self.prot_seq_lstm(prot_node_dense)
                    prot_graph_rep_diff = masked_mean(prot_lstm_out, mask=prot_node_mask, dim=1)

        chem_fused2, prot_fused2 = None, None
        if self.use_cross_attention:
            chem_fused2, prot_fused2 = self.cross_att(
                chem_graph_rep_diff_org[idx_drug], atom_pad_mask[idx_drug],
                prot_node_dense[idx_target], prot_node_mask[idx_target])

        aux = {}
        if return_aux_loss and self.use_cvcl:
            with torch.autocast(device_type='cuda', enabled=False):
                if self.use_comp_graph and self.use_comp_seq:
                    aux['chem_contrastive'] = layerwise_view_contrastive(
                        chem_graph_rep_diff.unsqueeze(1), chem_emb_all,
                        self.chem_graph_proj, self.chem_seq_proj, t=0.15)
                if self.use_prot_graph and self.use_prot_seq:
                    aux['prot_contrastive'] = layerwise_view_contrastive(
                        prot_graph_rep_diff.unsqueeze(1), prot_emb_all,
                        self.prot_graph_proj, self.prot_seq_proj, t=0.15)
                if self.use_cross_attention and self.use_comp_graph:
                    aux['prot_contrastive_2'] = layerwise_view_contrastive(
                        chem_emb_all[idx_drug].unsqueeze(1), chem_fused2,
                        self.chem_graph_proj, self.cross_graph_proj, t=0.15)
                if self.use_cross_attention and self.use_prot_graph:
                    aux['prot_contrastive_3'] = layerwise_view_contrastive(
                        prot_emb_all[idx_target].unsqueeze(1), prot_fused2,
                        self.prot_graph_proj, self.cross_seq_proj, t=0.15)

        features = []
        for name in self.branch_names:
            if   name == 'comp_graph':  features.append(chem_emb)
            elif name == 'prot_graph':  features.append(prot_emb)
            elif name == 'fp':          features.append(repr_fp[idx_drug])
            elif name == 'comp_seq':    features.append(chem_graph_rep_diff[idx_drug])
            elif name == 'prot_seq':    features.append(prot_graph_rep_diff[idx_target])
            elif name == 'cross_comp':  features.append(chem_fused2)
            elif name == 'cross_prot':  features.append(prot_fused2)

        combined = self.dynamic_gate(*features)
        y_final2 = self.seq_expert(combined).squeeze(-1)

        return y_final2, y_final2, aux