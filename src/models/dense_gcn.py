import torch
import torch.nn as nn
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import DenseGCNConv, GCNConv, global_mean_pool as gep


class DenseGCNBlock(nn.Module):
    def __init__(self, gcn_layers_dim, dropout_rate=0.,
                 relu_layers_index=(), dropout_layers_index=()):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        for i in range(len(gcn_layers_dim) - 1):
            self.conv_layers.append(DenseGCNConv(gcn_layers_dim[i], gcn_layers_dim[i + 1]))
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.relu_layers_index    = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x, adj):
        output, embeddings = x, []
        for i, conv in enumerate(self.conv_layers):
            output = conv(output, adj, add_loop=False)
            if i in self.relu_layers_index:
                output = self.relu(output)
            if i in self.dropout_layers_index:
                output = self.dropout(output)
            embeddings.append(torch.squeeze(output, dim=0))
        return embeddings


class DenseGCNModel(nn.Module):
    def __init__(self, layers_dim, edge_dropout_rate=0.):
        super().__init__()
        self.edge_dropout_rate = edge_dropout_rate
        self.num_layers = len(layers_dim) - 1
        self.graph_conv = DenseGCNBlock(
            layers_dim, 0.1,
            relu_layers_index=list(range(self.num_layers)),
            dropout_layers_index=list(range(self.num_layers)))

    def forward(self, graph):
        xs, adj = graph.x, graph.adj
        num_d, num_t = graph.num_drug, graph.num_target
        indexs = torch.where(adj != 0)
        edge_indexs = torch.cat((indexs[0].unsqueeze(0), indexs[1].unsqueeze(0)), 0)
        edge_indexs_dropout, edge_weights_dropout = dropout_adj(
            edge_index=edge_indexs, edge_attr=adj[indexs],
            p=self.edge_dropout_rate, force_undirected=True,
            num_nodes=num_d + num_t, training=self.training)
        adj_dropout = torch.zeros_like(adj)
        adj_dropout[edge_indexs_dropout[0], edge_indexs_dropout[1]] = edge_weights_dropout
        return self.graph_conv(xs, adj_dropout)