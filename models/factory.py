import torch
from models.gcn import GCN
from models.gineconv import GINE
from models.gat import GAT

MODEL_REGISTRY = {
    "gcn": GCN,
    "gineconv": GINE,
    "gat": GAT
}

def build_model(config, device, num_node_features, weight_seed=None):
    if weight_seed is not None:
        torch.manual_seed(weight_seed)

    model_type = config['model']['model_type']

    model_num_heads = config['model']['num_heads'] if model_type == 'gat' else None

    model_class = MODEL_REGISTRY.get(model_type)

    if model_class is None:
        raise ValueError(f"Unknown model_type: '{model_type}'. Available: {list(MODEL_REGISTRY.keys())}")
    
    else:
        model = model_class(num_node_features, num_layers=config['model']['num_layers'], hidden_dim=config['model']['hidden_dim'], heads=model_num_heads)

    return model.to(device) 