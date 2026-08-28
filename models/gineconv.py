import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool

class GINE(nn.Module):
    def __init__(self, num_node_features, num_layers, hidden_dim, edge_dim=4):
        super().__init__()
        conv_layers = []

        mlp1 = nn.Sequential(nn.Linear(num_node_features, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        conv_layers.append(GINEConv(nn=mlp1, edge_dim=edge_dim))

        for _ in range(num_layers - 1):
            mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            conv_layers.append(GINEConv(nn=mlp, edge_dim=edge_dim))

        self.convs = nn.ModuleList(conv_layers)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        out = x

        for conv in self.convs:
            out = conv(out, edge_index, edge_attr)
            out = F.relu(out)

        out = F.dropout(out, p=0.2, training=self.training)
        out = global_mean_pool(out, batch)
        out = self.linear(out)
        
        return out

