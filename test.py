import torch
import argparse
from utils import print_config, set_seed
from training.loop import validate
from evaluation.load_model import load_model_from_checkpoint

def test_pipeline(model_path):
    bundle = load_model_from_checkpoint(model_path)
    
    if bundle['config']['logging']['config_debug']:
        print_config(bundle['config'])

    # From the bundle, use config -> logging options -> seed
    set_seed(bundle['config']['data']['seed'])

    criterion = torch.nn.L1Loss()
    test_loss = validate(bundle['model'], bundle['test_loader'], criterion, 
        bundle['device'], bundle['mean'], bundle['std'], bundle['config']['data']['target_idx'])

    print(f"Test loss: {test_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="Path to the saved model checkpoint")
    args = parser.parse_args()

    test_pipeline(args.model_path)