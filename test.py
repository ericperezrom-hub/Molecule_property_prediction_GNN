import torch
import argparse
from utils import print_config
from dataloader import get_dataloader
from train import validate
from models.baseline import GCN
from utils import set_seed

def load_model_from_checkpoint(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(model_path, map_location=device)

    train_config = checkpoint['config']

    model = GCN(num_node_features=11, hidden_dim=train_config['hidden_dim']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    mean = checkpoint['mean'].to(device)
    std = checkpoint['std'].to(device)

    return {
        'model': model,
        'mean': mean,
        'std': std,
        'device': device,
        'config': train_config
    }

def test_pipeline(model_path):
    bundle = load_model_from_checkpoint(model_path)
    
    if bundle['config']['config_debug']:
        print_config(bundle['config'])

    set_seed(bundle['config']['seed'])
    _, _, test_loader, _ = get_dataloader(bundle['config'])

    criterion = torch.nn.L1Loss()
    test_loss = validate(bundle['model'], test_loader, criterion, bundle['device'], bundle['mean'], bundle['std'], bundle['config']['target_idx'])
    print(f"Test loss: {test_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="Path to the saved model checkpoint")
    args = parser.parse_args()

    test_pipeline(args.model_path)