import torch
from models.factory import build_model

CRITERION_REGISTRY = {
    "l1": torch.nn.L1Loss,
    "mse": torch.nn.MSELoss,
}

OPTIMIZER_REGISTRY = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
}

def build_training_components(config, device, num_node_features):
    model = build_model(
        config,
        device,
        num_node_features=num_node_features,
    )

    optimizer_class = OPTIMIZER_REGISTRY.get(config['optimizer_type'])
    if optimizer_class is None:
        raise ValueError(f"Unknown optimizer_type: '{config['optimizer_type']}'. Available: {list(OPTIMIZER_REGISTRY.keys())}")
    optimizer = optimizer_class(model.parameters(), lr=config['lr'])

    criterion_class = CRITERION_REGISTRY.get(config['criterion_type'])
    if criterion_class is None:
        raise ValueError(f"Unknown criterion_type: '{config['criterion_type']}'. Available: {list(CRITERION_REGISTRY.keys())}")
    criterion = criterion_class()

    return model, optimizer, criterion