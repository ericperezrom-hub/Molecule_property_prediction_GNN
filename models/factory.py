import torch
from models.gcn import GCN

MODEL_REGISTRY = {
    "gcn": GCN
}

def build_model(config, device, num_node_features, weight_seed=None):
    if weight_seed is not None:
        torch.manual_seed(weight_seed)

    model_type = config['model']['model_type']

    model_class = MODEL_REGISTRY.get(model_type)

    if model_class is None:
        raise ValueError(f"Unknown model_type: '{model_type}'. Available: {list(MODEL_REGISTRY.keys())}")
    
    else:
        model = model_class(num_node_features, config['model']['num_layers'], config['model']['hidden_dim'])

    return model.to(device)