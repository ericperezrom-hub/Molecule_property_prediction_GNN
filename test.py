import torch
import wandb
import argparse
from utils import print_config
from training.loop import validate
from evaluation.load_model import load_model_from_checkpoint
from evaluation.test_utils import compute_trivial_baseline, log_test_results

def test_pipeline(model_path):
    bundle = load_model_from_checkpoint(model_path)
    
    if bundle['config']['logging']['config_debug']:
        print_config(bundle['config'])

    criterion = torch.nn.L1Loss()

    run = wandb.init(project="gnn-molecule-prediction-test", config=bundle['config'])

    test_loss = validate(bundle['model'], bundle['test_loader'], criterion, 
        bundle['device'], bundle['mean'], bundle['std'], bundle['config']['data']['target_idx'])

    baseline_error = compute_trivial_baseline(
        bundle['test_loader'].dataset,
        bundle['mean'],
        bundle['config']['data']['target_idx'],
        bundle['device']
)

    print(f"Test loss: {test_loss:.4f}")
    print(f"Baseline error: {baseline_error:.4f}")

    log_test_results(test_loss, baseline_error)
    
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="Path to the saved model checkpoint")
    args = parser.parse_args()

    test_pipeline(args.model_path)