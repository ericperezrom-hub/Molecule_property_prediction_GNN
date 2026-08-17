import torch
from models.factory import build_model
from dataloader import get_dataloader

def load_model_from_checkpoint(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(model_path, map_location=device)

    train_config = checkpoint['config']

    _, _, test_loader, _, test_data, num_node_features = get_dataloader(train_config)

    model = build_model(train_config, device, num_node_features=num_node_features)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    mean = checkpoint['mean'].to(device)
    std = checkpoint['std'].to(device)

    return {
        'model': model,
        'mean': mean,
        'std': std,
        'device': device,
        'config': train_config,
        'test_loader': test_loader,
        'test_data': test_data
    }