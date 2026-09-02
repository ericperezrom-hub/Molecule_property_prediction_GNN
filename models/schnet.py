import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models.schnet import InteractionBlock, GaussianSmearing
from torch_geometric.nn import global_mean_pool

def radius_graph_batch(pos, batch, r):
    dist = torch.cdist(pos, pos)
    same_graph = batch.unsqueeze(0) == batch.unsqueeze(1)
    adj = same_graph & (dist <= r) & (dist > 0)
    row, col = adj.nonzero(as_tuple=True)
    return torch.stack([row, col], dim=0)


class MoleculeSchNetManual(nn.Module):
    def __init__(self, num_layers, hidden_dim, cutoff=10.0, num_gaussians=50):
        super().__init__()
        self.embedding = nn.Embedding(100, hidden_dim)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)
        self.cutoff = cutoff

        self.interactions = nn.ModuleList([
            InteractionBlock(hidden_dim, num_gaussians, hidden_dim, cutoff)
            for _ in range(num_layers)
        ])

        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        z, pos, batch = data.z, data.pos, data.batch

        edge_index = radius_graph_batch(pos, batch, self.cutoff)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)

        out = self.embedding(z)
        for interaction in self.interactions:
            out = out + interaction(out, edge_index, edge_weight, edge_attr)

        out = F.dropout(out, p=0.2, training=self.training)
        out = global_mean_pool(out, batch)
        out = self.linear(out)

        return out