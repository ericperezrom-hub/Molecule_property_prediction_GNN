import torch
from models.gcn import GCN
from models.gineconv import GINE
from models.gat import GAT
from models.nnconv import MoleculeNNConv
from models.schnet import MoleculeSchNetManual

MODEL_REGISTRY = {
    "gcn": GCN,
    "gineconv": GINE,
    "gat": GAT,
    "nnconv": MoleculeNNConv,
    "schnet": MoleculeSchNetManual,
}

def build_model(config, device, num_node_features, weight_seed=None):

    if weight_seed is not None:
        torch.manual_seed(weight_seed)

    model_type = config['model']['model_type']

    model_class = MODEL_REGISTRY.get(model_type)

    if model_class is None:
        raise ValueError(
            f"Unknown model_type: '{model_type}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    model_kwargs = {
        "num_layers": config['model']['num_layers'],
        "hidden_dim": config['model']['hidden_dim'],
    }

    if model_type == "gat":
        model_kwargs["heads"] = config['model']['num_heads']

    if model_type == "schnet":
        model_kwargs["cutoff"] = config['model'].get('cutoff', 10.0)
        model_kwargs["num_gaussians"] = config['model'].get('num_gaussians', 50)
        model = model_class(**model_kwargs)
    else:
        model = model_class(num_node_features, **model_kwargs)

    return model.to(device)