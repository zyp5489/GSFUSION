import pickle
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from .esm import ESM_FEAT_DIR, load_esm_features


class pygFpDataSet(Dataset):
    def __init__(self, dataset, root_dir,
                 max_ligand_graphs=1024,
                 max_protein_graphs=1024):
        super().__init__()
        self.dataset = dataset
        self.max_ligand_graphs  = max_ligand_graphs
        self.max_protein_graphs = max_protein_graphs
        self.max_seq_len = 128

        # ---- 蛋白图 + ESM 回填 ----
        self.protein_graph = pickle.load(
            open(f"{root_dir}/davis_protein_graphs_with_esm6.pkl", "rb"))
        print("protein graph num:", len(self.protein_graph))

        loaded, legacy, skipped = 0, 0, 0
        for data in self.protein_graph:
            x2 = None
            if hasattr(data, 'x2') and data.x2 is not None:
                x2 = data.x2.float(); legacy += 1
            elif getattr(data, 'esm_file', None) is not None:
                try:
                    x2 = load_esm_features(data, esm_feat_dir=ESM_FEAT_DIR).float()
                    loaded += 1
                except Exception as e:
                    print(f"⚠️ 加载 ESM 失败 {getattr(data, 'upacc', '?')}: {e}")
                    skipped += 1
            else:
                print(f"⚠️ 图缺少 ESM 信息: upacc={getattr(data, 'upacc', '?')}")
                skipped += 1

            if x2 is not None:
                data.x2 = x2
            for attr in ('esm_file', 'esm_index', 'upacc'):
                try:
                    if attr in data:
                        del data[attr]
                except Exception:
                    pass
        print(f"ESM 回填完成: loaded={loaded}, legacy={legacy}, skipped={skipped}")

        # ---- 配体 pair / fingerprint ----
        self.ligand_pair = pickle.load(
            open(f"{root_dir}/davis_drug_graph_molformer_pair.pkl", "rb"))
        print("ligand pair num:", len(self.ligand_pair))

        self.ligand_fp = pickle.load(
            open(f"{root_dir}/davis_ligand_fps.pkl", "rb"))
        print("ligand fp num:", len(self.ligand_fp))

        # ---- 预 batch ----
        self.batch_ligand_graph  = Batch.from_data_list(
            [self.ligand_pair[t][0] for t in range(len(self.ligand_pair))])
        self.batch_protein_graph = Batch.from_data_list(self.protein_graph)
        self.batch_ligand_fp     = torch.stack(self.ligand_fp)

        # ---- token 预 padding ----
        tokens = [self.ligand_pair[t][1] for t in range(len(self.ligand_pair))]
        max_seq_len = min(max(t.shape[0] for t in tokens), self.max_seq_len)
        token_dim   = tokens[0].shape[1]
        n = len(tokens)
        self.batch_ligand_token = torch.zeros(n, max_seq_len, token_dim)
        self.atom_pad_mask      = torch.zeros(n, max_seq_len, dtype=torch.bool)
        for i, t in enumerate(tokens):
            n_atom = min(t.shape[0], max_seq_len)
            self.batch_ligand_token[i, :n_atom, :] = torch.from_numpy(t).float()[:n_atom, :]
            self.atom_pad_mask[i, :n_atom]         = True

        self.num_ligand_graphs  = self.batch_ligand_graph.num_graphs
        self.num_protein_graphs = self.batch_protein_graph.num_graphs
        print(f"全量配体图数: {self.num_ligand_graphs}, "
              f"每步配体图池上限 N_l: {self.max_ligand_graphs}")
        print(f"全量蛋白图数: {self.num_protein_graphs}, "
              f"每步蛋白图池上限 N_p: {self.max_protein_graphs}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    @staticmethod
    def _take_pyg_batch(big_batch, indices):
        data_list = [big_batch[int(i)] for i in indices.tolist()]
        return Batch.from_data_list(data_list)

    @staticmethod
    def _build_pool(all_ids_used, total, cap):
        num_unique = all_ids_used.numel()
        if num_unique >= cap:
            return all_ids_used
        mask = torch.ones(total, dtype=torch.bool)
        mask[all_ids_used] = False
        candidates = torch.arange(total)[mask]
        need = cap - num_unique
        if candidates.numel() > need:
            perm = torch.randperm(candidates.numel())[:need]
            extra_ids = candidates[perm]
        else:
            extra_ids = candidates
        return torch.cat([all_ids_used, extra_ids])

    def collate_fn(self, batch):
        ys, drug_ids, target_ids = [], [], []
        for ligand_index, protein_index, y in batch:
            ys.append(y); drug_ids.append(ligand_index); target_ids.append(protein_index)

        ys         = torch.tensor(ys, dtype=torch.float)
        drug_ids   = torch.tensor(drug_ids,   dtype=torch.long)
        target_ids = torch.tensor(target_ids, dtype=torch.long)

        unique_drug_ids, drug_inverse = torch.unique(drug_ids, return_inverse=True)
        sel_lig = self._build_pool(unique_drug_ids, self.num_ligand_graphs,
                                   self.max_ligand_graphs)

        batch_ligand_graph = self._take_pyg_batch(self.batch_ligand_graph, sel_lig)
        batch_ligand_token = self.batch_ligand_token[sel_lig].clone()
        atom_pad_mask      = self.atom_pad_mask[sel_lig].clone()
        batch_ligand_fp    = self.batch_ligand_fp[sel_lig].clone()

        unique_prot_ids, prot_inverse = torch.unique(target_ids, return_inverse=True)
        sel_prot = self._build_pool(unique_prot_ids, self.num_protein_graphs,
                                    self.max_protein_graphs)
        batch_protein_graph = self._take_pyg_batch(self.batch_protein_graph, sel_prot)

        return (ys, batch_ligand_fp, batch_ligand_graph, batch_protein_graph,
                batch_ligand_token, atom_pad_mask, drug_inverse, prot_inverse)