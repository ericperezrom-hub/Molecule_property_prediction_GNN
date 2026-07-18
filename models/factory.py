from models.gcn import GCN

MODEL_REGISTRY = {
    "gcn": GCN
}

def build_model(config, device, num_node_features):
    model_type = config['model_type']

    model_class = MODEL_REGISTRY.get(model_type)

    if model_class is None:
        raise ValueError(f"Unknown model_type: '{model_type}'. Available: {list(MODEL_REGISTRY.keys())}")
    
    else:
        model = model_class(num_node_features, config['num_layers'], config['hidden_dim'])

    return model.to(device)