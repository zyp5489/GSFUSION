import json
import pickle
import numpy as np
import scipy.sparse as sp
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch_geometric import data as DATA

from .dataset import pygFpDataSet
from .utils   import (minMaxNormalize, denseAffinityRefine,
                      sparse_mx_to_torch_sparse_tensor)


class testFpDataModule(pl.LightningDataModule):
    def __init__(self, **args):
        super().__init__()
        self.dataset_name = args["dataset_name"]
        self.batch_size   = args["batch_size"]
        self.train_fold_index = 0
        self.full_affinity = None
        self.args = args

    # -------- 读亲和力矩阵 --------
    def read_data(self, dataset):
        dataset_path = 'data/' + dataset + '/'
        affinity = pickle.load(open(dataset_path + 'davis_affinities', 'rb'),
                               encoding='latin1')
        if dataset in ('davis', 'davis_cold_target', 'davis_cold_drug', 'davis_cold'):
            affinity = [-np.log10(y / 1e9) for y in affinity]
        return np.asarray(affinity)

    # -------- 构造亲和力图 / 正样本掩码 --------
    def get_affinity_graph(self, dataset, adj, num_pos, pos_threshold):
        dataset_path = 'data/' + dataset + '/'
        num_drug, num_target = adj.shape

        # 药-药正样本
        dt_ = adj.copy()
        dt_ = np.where(dt_ >= pos_threshold, 1.0, 0.0)
        dtd = np.matmul(dt_, dt_.T)
        dtd = dtd / dtd.sum(axis=-1).reshape(-1, 1)
        dtd = np.nan_to_num(dtd)
        dtd += np.eye(num_drug, num_drug)
        dtd = dtd.astype("float32")

        drug_pos = np.zeros((num_drug, num_drug))
        for i in range(len(dtd)):
            one = dtd[i].nonzero()[0]
            if len(one) > num_pos:
                sele = one[np.argsort(-dtd[i, one])[:num_pos]]
                drug_pos[i, sele] = 1
            else:
                drug_pos[i, one] = 1
        drug_pos = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(drug_pos))

        # 靶-靶正样本
        td_ = adj.T.copy()
        td_ = np.where(td_ >= pos_threshold, 1.0, 0.0)
        tdt = np.matmul(td_, td_.T)
        tdt = tdt / tdt.sum(axis=-1).reshape(-1, 1)
        tdt = np.nan_to_num(tdt)
        tdt += np.eye(num_target, num_target)
        tdt = tdt.astype("float32")

        target_pos = np.zeros((num_target, num_target))
        for i in range(len(tdt)):
            one = tdt[i].nonzero()[0]
            if len(one) > num_pos:
                sele = one[np.argsort(-tdt[i, one])[:num_pos]]
                target_pos[i, sele] = 1
            else:
                target_pos[i, one] = 1
        target_pos = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(target_pos))

        # 归一化 + 大图
        if dataset in ('davis', 'davis_cold_target', 'davis_cold_drug', 'davis_cold', 'bindingdb'):
            adj[adj != 0] -= 5
            adj_norm = minMaxNormalize(adj, 0)
        elif dataset in ('kiba', 'metz'):
            adj_refine = denseAffinityRefine(adj.T, 150)
            adj_refine = denseAffinityRefine(adj_refine.T, 40)
            adj_norm = minMaxNormalize(adj_refine, 0)
        else:
            raise ValueError(f"未知 dataset: {dataset}")

        adj_1, adj_2 = adj_norm, adj_norm.T
        adj = np.concatenate((
            np.concatenate((np.zeros([num_drug, num_drug]), adj_1), 1),
            np.concatenate((adj_2, np.zeros([num_target, num_target])), 1)), 0)

        train_row_ids, train_col_ids = np.where(adj != 0)
        edge_indexs = np.stack([train_row_ids, train_col_ids], 0)
        edge_weights = adj[train_row_ids, train_col_ids]

        node_type_features = np.concatenate((
            np.tile(np.array([1, 0]), (num_drug, 1)),
            np.tile(np.array([0, 1]), (num_target, 1))), 0)
        adj_features = np.zeros_like(adj)
        adj_features[adj != 0] = 1
        features = np.concatenate((node_type_features, adj_features), 1)

        affinity_graph = DATA.Data(x=torch.Tensor(features),
                                   adj=torch.Tensor(adj),
                                   edge_index=torch.LongTensor(edge_indexs))
        affinity_graph.__setitem__("edge_weight", torch.Tensor(edge_weights))
        affinity_graph.__setitem__("num_drug", num_drug)
        affinity_graph.__setitem__("num_target", num_target)
        return affinity_graph, drug_pos, target_pos

    def process_data(self, affinity_mat, dataset, num_pos, pos_threshold):
        dataset_path = 'data/' + dataset + '/'
        train_file = json.load(open(dataset_path + '/folds/train_fold_setting1.txt'))
        train_index = []
        for fold in train_file:
            train_index += fold

        rows, cols = np.where(np.isnan(affinity_mat) == False)
        train_rows, train_cols = rows[train_index], cols[train_index]
        train_Y = affinity_mat[train_rows, train_cols]

        train_affinity_mat = np.zeros_like(affinity_mat)
        train_affinity_mat[train_rows, train_cols] = train_Y

        affinity_graph, drug_pos, target_pos = self.get_affinity_graph(
            dataset, train_affinity_mat, num_pos, pos_threshold)
        return affinity_graph, drug_pos, target_pos

    # -------- setup --------
    def setup(self, stage=None):
        dataset_name = self.args["dataset_name"]
        self.dataset = pickle.load(
            open(f'./data/{dataset_name}/DTAs_new.pkl', 'rb'))

        root_dir = f'./data/{self.dataset_name}'
        train_fold = json.load(open(f"data/{dataset_name}/folds/train_fold_setting1.txt"))
        test_fold  = json.load(open(f"data/{dataset_name}/folds/test_fold_setting1.txt"))

        trainset_fold = [self.dataset[idx] for fold in range(len(train_fold)) for idx in train_fold[fold]]
        testset_fold  = [self.dataset[idx] for idx in test_fold]

        self.train_dataset = pygFpDataSet(trainset_fold, root_dir)
        self.test_dataset  = pygFpDataSet(testset_fold,  root_dir)

        affinity = self.read_data(dataset_name)

        train_folds = []
        for fold in train_fold:
            train_folds += fold
        rows, cols = np.where(np.isnan(affinity) == False)
        train_rows, train_cols = rows[train_folds], cols[train_folds]
        train_Y = affinity[train_rows, train_cols]

        affinity_train = np.zeros_like(affinity)
        affinity_train[train_rows, train_cols] = train_Y

        num_pos       = 10
        pos_threshold = 8.0
        affinity_graph, drug_pos, target_pos = self.process_data(
            affinity_train, dataset_name, num_pos, pos_threshold)
        self.full_affinity = affinity_graph
        self.drug_pos      = drug_pos
        self.target_pos    = target_pos

    # -------- dataloaders --------
    def train_dataloader(self):
        return DataLoader(self.train_dataset, collate_fn=self.train_dataset.collate_fn,
                          batch_size=self.batch_size, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, collate_fn=self.test_dataset.collate_fn,
                          batch_size=self.batch_size, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, collate_fn=self.test_dataset.collate_fn,
                          batch_size=self.batch_size,
                          pin_memory=torch.cuda.is_available())