import numpy as np
import scipy.sparse as sp
import torch


def minMaxNormalize(Y, Y_min=None, Y_max=None):
    if Y_min is None: Y_min = np.min(Y)
    if Y_max is None: Y_max = np.max(Y)
    return (Y - Y_min) / (Y_max - Y_min)


def denseAffinityRefine(adj, k):
    refine_adj = np.zeros_like(adj)
    indexs1 = np.tile(np.expand_dims(np.arange(adj.shape[0]), 0), (k, 1)).transpose()
    indexs2 = np.argpartition(adj, -k, 1)[:, -k:]
    refine_adj[indexs1, indexs2] = adj[indexs1, indexs2]
    return refine_adj


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape  = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)