import torch
import argparse
from utils import load_config
from dataloader import get_dataloader
from train import validate
from models.baseline import GCN
from utils import set_seed

def test_pipeline(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(model_path, map_location=device)

    train_config = checkpoint['config']
    set_seed(train_config['seed'])

    _, _, test_loader, _ = get_dataloader(train_config)

    model = GCN(num_node_features=11, hidden_dim=train_config['hidden_dim']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    mean = checkpoint['mean']
    std = checkpoint['std']

    criterion = torch.nn.L1Loss()

    test_loss = validate(model, test_loader, criterion, device, mean, std, train_config['target_idx'])
    print(f"Test loss: {test_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="Path to the saved model checkpoint")
    args = parser.parse_args()

    test_pipeline(args.model_path)