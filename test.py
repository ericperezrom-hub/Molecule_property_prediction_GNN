import torch
import wandb
import argparse
from utils import load_config
from dataloader import get_dataloader
from train import validate
from models.baseline import GCN

def test_pipeline(config, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, _, test_loader, _ = get_dataloader(config)

    checkpoint = torch.load(model_path, weights_only=True)

    model = GCN(num_node_features=11, hidden_dim=config['hidden_dim']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    mean = checkpoint['mean']
    std = checkpoint['std']

    criterion = torch.nn.L1Loss()

    test_loss = validate(model, test_loader, criterion, device, mean, std, config['target_idx'])
    print(f"Test loss: {test_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="Path to the saved model checkpoint")
    args = parser.parse_args()

    config = load_config('config.yaml')
    test_pipeline(config, args.model_path)