from collections import *
import torch
from torch import nn
from torch.nn import functional as F
from models.equiv_layers import *
from utils.graph_utils import *


class GraphIsomorphismNetworkLayer(nn.Module):
    def __init__(self, input_dim, output_dim, activation, batch_norm):
        super(GraphIsomorphismNetworkLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = False

        if activation is not None:
            self.activation = getattr(F, activation)
        else:
            self.activation = None
        return

    def forward(self, x):
        x = self.linear(x)
        B, N, d = x.size()
        x = x.reshape(B*N, d)
        if self.batch_norm:
            x = self.batch_norm(x)
        x = x.reshape(B, N, d)
        if self.activation:
            x = self.activation(x)
        return x


class GraphIsomorphismNetwork(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, output_dim, activation, epsilon, batch_norm=True, device = "cpu"):
        super(GraphIsomorphismNetwork, self).__init__()
        self.gin_layers_dim = [node_feature_dim] + hidden_dim
        self.gin_layers_num = len(self.gin_layers_dim)
        self.epsilon = epsilon
        self.output_dim = output_dim
        self.batch_norm = batch_norm

        self.gin_layers = nn.ModuleList()
        self.combine_net = nn.ModuleList()
        for layer_idx, (in_dim, out_dim) in enumerate(zip(self.gin_layers_dim[:-1], self.gin_layers_dim[1:])):
            self.gin_layers.append(GraphIsomorphismNetworkLayer(in_dim, out_dim, activation, batch_norm))
            self.combine_net.append(nn.Sequential(nn.Linear(out_dim * 2, out_dim), nn.LeakyReLU(), nn.Linear(out_dim, out_dim)))
        self.fdim = hidden_dim[-1]
        self.lin_edge = nn.Sequential(nn.Linear(4, hidden_dim[-1] * 2), nn.Tanh(), nn.Linear(hidden_dim[-1] * 2, hidden_dim[-1]))
        return

    def represent(self,adjacency, x, edge_feat = None):
        edge_feat = self.lin_edge(edge_feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        for layer_idx in range(self.gin_layers_num-1):
            x = (1 + self.epsilon) * x + torch.einsum("bnn,bnd -> bnd", adjacency, x)
            x = self.gin_layers[layer_idx](x)
            x = torch.cat([torch.einsum("bcij, bjk->bik", edge_feat, x), x], dim = -1)
            x = self.combine_net[layer_idx](x)
        return x, edge_feat

    def forward(self, adjacency, x, edge_feat, mask):
        x, adj = self.represent(adjacency, x, edge_feat)
        return x.transpose(1, 2), adj

if __name__ == "__main__":
    x = torch.randn(4, 38, 9)
    adj = torch.randn(4, 4, 38, 38)
    net = GraphIsomorphismNetwork(9, [128, 128], 128, "relu", 1e-6)
    out = net(x, adj)
    print(out.shape)