import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GCN(nn.Module):
    def __init__(self, num_node_features, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, 1)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        out = self.conv1(x, edge_index)
        out = F.relu(out)
        out = self.conv2(out, edge_index)
        out = F.relu(out)
        out = F.dropout(out, p=0.2, training=self.training)
        out = global_mean_pool(out, batch)
        out = self.linear(out)
        return out