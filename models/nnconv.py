import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import NNConv as PyGNNConv
from torch_geometric.nn import global_mean_pool

class MoleculeNNConv(nn.Module):
    def __init__(self, num_node_features, num_layers, hidden_dim, edge_dim=4, edge_network_hidden=32):
        super().__init__()
        conv_layers = []

        edge_network = nn.Sequential(
            nn.Linear(edge_dim, edge_network_hidden),
            nn.ReLU(),
            nn.Linear(edge_network_hidden, num_node_features * hidden_dim)
        )

        layer_1 = PyGNNConv(num_node_features, hidden_dim, edge_network, aggr='mean')
        conv_layers.append(layer_1)

        for _ in range(num_layers - 1):
            edge_network = nn.Sequential(
                nn.Linear(edge_dim, edge_network_hidden),
                nn.ReLU(),
                nn.Linear(edge_network_hidden, hidden_dim * hidden_dim)
            )

            conv_layers.append(PyGNNConv(hidden_dim, hidden_dim, edge_network, aggr='mean'))

        self.convs = nn.ModuleList(conv_layers)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, data):

        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        out = x

        for conv in self.convs:
            out = conv(out, edge_index, edge_attr)
            out = F.relu(out)

        out = F.dropout(out, p=0.2,training=self.training)
        out = global_mean_pool(out, batch)
        out = self.linear(out)

        return out