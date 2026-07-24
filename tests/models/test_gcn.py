import torch
from torch_geometric.data import Data
from models.gcn import GCN

def test_gcn_forward_pass():
    num_node_features = 3
    hidden_dim = 8
    num_layers = 2

    model = GCN(num_node_features, num_layers, hidden_dim)
    model.eval()

    x = torch.randn((4, num_node_features))
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    y = torch.randn((2, 1))

    data = Data(x=x, edge_index=edge_index, batch=batch, y=y)

    out = model(data)

    assert out.shape == (2, 1)
    assert torch.isfinite(out).all()
