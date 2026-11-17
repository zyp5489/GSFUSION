import torch.nn as nn

param_dict = {
    "chem_in_features": 34,
    "prot_in_features": 39,
    "hidden_dim": 256,
    "chem_heads_out_feat_params": [32] * 6,
    "chem_blocks_params":         [8]  * 6,
    "dropout_1": 0.4,
    "dropout_2": 0.1,
    "dropout_3": 0.5,
    "prot_heads_out_feat_params": [32] * 4,
    "prot_blocks_params":         [8]  * 4,
    "prot_windows_params":        [7]  * 4,
    "batch_size": 512,
    "lr": 5e-4,
    "dataset_name": "davis",
    "criterion": nn.MSELoss(),
    "model_name": "ssr-dta",
    "max_seq_len": 2000,
    "n_embd": 256,
    "cross_pos_bias_switch": False,
    "masking": False,
    "contrast_weight": 1,
    "return_contrast_loss": False,
    "using_lstm": True,

    # 消融开关
    'use_prot_seq': True,
    'use_comp_seq': True,
    'use_prot_graph': True,
    'use_comp_graph': False,
    'use_cross_attention': True,
    'use_gcgim': True,
    'use_grc': True,
    'use_cvcl': True,

    "L_view_contrastive_weight": 1.0,
}