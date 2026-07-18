import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GCN(nn.Module):
    def __init__(self, num_node_features, num_layers, hidden_dim):
        super().__init__()
        conv_layers = []
        layer_1 = GCNConv(num_node_features, hidden_dim)
        conv_layers.append(layer_1)
        for _ in range(num_layers - 1):
            layer = GCNConv(hidden_dim, hidden_dim)
            conv_layers.append(layer)

        self.convs = nn.ModuleList(conv_layers)
        self.linear = nn.Linear(hidden_dim, 1)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        out = x

        for conv in self.convs:
            out = conv(out, edge_index)
            out = F.relu(out)

        out = F.dropout(out, p=0.2, training=self.training)
        out = global_mean_pool(out, batch)
        out = self.linear(out)
        
        return out