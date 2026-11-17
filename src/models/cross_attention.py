import torch
import torch.nn as nn


class cross_AFT(nn.Module):
    def __init__(self, config, drug_dim, protein_dim):
        super().__init__()
        self.config = config

        self.pos_bias = nn.Parameter(
            torch.zeros(config['max_seq_len'], config['max_seq_len']),
            requires_grad=config['cross_pos_bias_switch'])
        self.Activation = nn.Sigmoid()

        self.query_d  = nn.Linear(drug_dim,    config['n_embd'])
        self.key_p    = nn.Linear(protein_dim, config['n_embd'])
        self.value_p  = nn.Linear(protein_dim, config['n_embd'])

        self.query_p  = nn.Linear(protein_dim, config['n_embd'])
        self.key_d    = nn.Linear(drug_dim,    config['n_embd'])
        self.value_d  = nn.Linear(drug_dim,    config['n_embd'])

        self.proj_d   = nn.Linear(config['n_embd'], config['n_embd'])
        self.proj_p   = nn.Linear(config['n_embd'], config['n_embd'])

    def forward(self, smiles_embeddings, smiles_mask,
                protein_embeddings, protein_mask):
        B_d, T_d, _ = smiles_embeddings.size()
        B_p, T_p, _ = protein_embeddings.size()
        assert B_d == B_p

        smiles_embeddings  = smiles_embeddings.permute(1, 0, 2)   # (T_d, B, C)
        protein_embeddings = protein_embeddings.permute(1, 0, 2)  # (T_p, B, C)

        if self.config['masking']:
            combined_mask = (smiles_mask.unsqueeze(2) * protein_mask.unsqueeze(1))  # (B,T_d,T_p)
            combined_mask = combined_mask.permute(1, 2, 0)                          # (T_d,T_p,B)

        # ---- drug 视角 ----
        q_d = self.query_d(smiles_embeddings)
        k_p = self.key_p(protein_embeddings)
        v_p = self.value_p(protein_embeddings)

        pos_bias = self.pos_bias[:T_d, :T_p].unsqueeze(-1)
        pos_bias_d = pos_bias.masked_fill(combined_mask == 0, -1e9) if self.config['masking'] else pos_bias

        max_key_p      = k_p.max(dim=0, keepdims=True)[0]
        max_pos_bias_d = pos_bias_d.max(dim=1, keepdims=True)[0]

        exp_key_p      = torch.exp(k_p - max_key_p)
        exp_pos_bias_d = torch.exp(pos_bias_d - max_pos_bias_d)

        num_d = torch.einsum('ijb,jbd->ibd', exp_pos_bias_d, exp_key_p * v_p)
        den_d = torch.einsum('ijb,jbd->ibd', exp_pos_bias_d, exp_key_p)

        y_d = self.Activation(q_d) * num_d / den_d
        y_d = self.proj_d(y_d).permute(1, 0, 2)   # (B, T_d, C)

        # ---- protein 视角 ----
        q_p = self.query_p(protein_embeddings)
        k_d = self.key_d(smiles_embeddings)
        v_d = self.value_d(smiles_embeddings)

        pos_bias = self.pos_bias[:T_d, :T_p].transpose(0, 1).unsqueeze(-1)
        pos_bias_p = pos_bias.masked_fill(combined_mask.permute(1, 0, 2) == 0, -1e9) \
            if self.config['masking'] else pos_bias

        max_key_d      = k_d.max(dim=0, keepdims=True)[0]
        max_pos_bias_p = pos_bias_p.max(dim=1, keepdims=True)[0]

        exp_key_d      = torch.exp(k_d - max_key_d)
        exp_pos_bias_p = torch.exp(pos_bias_p - max_pos_bias_p)

        num_p = torch.einsum('ijb,jbd->ibd', exp_pos_bias_p, exp_key_d * v_d)
        den_p = torch.einsum('ijb,jbd->ibd', exp_pos_bias_p, exp_key_d)

        y_p = self.Activation(q_p) * num_p / den_p
        y_p = self.proj_p(y_p).permute(1, 0, 2)

        if self.config['masking']:
            y_d = y_d.masked_fill(smiles_mask.unsqueeze(-1) == 0, -1e9)
            y_p = y_p.masked_fill(protein_mask.unsqueeze(-1) == 0, -1e9)

        y_d, _ = torch.max(y_d, dim=1)
        y_p, _ = torch.max(y_p, dim=1)
        return y_d, y_p