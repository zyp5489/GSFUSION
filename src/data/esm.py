import os
import torch


ESM_FEAT_DIR = os.environ.get("ESM_FEAT_DIR", "./esm_feats")
os.makedirs(ESM_FEAT_DIR, exist_ok=True)


def _torch_load(path, map_location='cpu'):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_esm_features(data, esm_feat_dir: str = ESM_FEAT_DIR) -> torch.Tensor:
    esm_file = getattr(data, 'esm_file', None)
    if esm_file is None:
        raise ValueError(
            f"data 缺少 esm_file 属性（upacc={getattr(data, 'upacc', '?')}）")

    path = os.path.join(esm_feat_dir, esm_file)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"ESM 特征文件缺失: {path}\n  upacc={getattr(data, 'upacc', '?')}")

    all_esm = _torch_load(path, map_location='cpu')
    if not torch.is_tensor(all_esm):
        all_esm = torch.as_tensor(all_esm)
    if all_esm.dtype != torch.float32:
        all_esm = all_esm.float()

    esm_index = getattr(data, 'esm_index', None)
    if esm_index is None:
        return all_esm

    esm_index = (torch.as_tensor(esm_index, dtype=torch.long)
                 if not torch.is_tensor(esm_index) else esm_index.long())

    n_rows = all_esm.shape[0]
    if esm_index.numel() > 0:
        min_idx = int(esm_index.min().item())
        max_idx = int(esm_index.max().item())
        if min_idx < 0 or max_idx >= n_rows:
            raise IndexError(
                f"esm_index 越界: min={min_idx}, max={max_idx}, "
                f"file rows={n_rows} (file={path})")

    return all_esm[esm_index]