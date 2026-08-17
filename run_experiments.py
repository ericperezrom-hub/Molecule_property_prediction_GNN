import copy
import json
from train import train_pipeline
from utils import load_config, validate_config

def run_multiple(base_config, n_epochs, n_runs=5):
    results = []

    for weight_seed in range(n_runs):
        config = copy.deepcopy(base_config)
        config['training']['n_epochs'] = n_epochs

        model_path, best_val_loss = train_pipeline(config, weight_seed=weight_seed)

        results.append({
            'weight_seed': weight_seed,
            'model_path': model_path,
            'best_val_loss': best_val_loss,
        })

    return results


if __name__ == "__main__":
    base_config = load_config('config.yaml')
    validate_config(base_config)

    print("Running 5x 50-epoch trainings...")
    results_50 = run_multiple(base_config, n_epochs=50, n_runs=5)

    print("Running 5x 200-epoch trainings...")
    results_200 = run_multiple(base_config, n_epochs=200, n_runs=5)

    with open('experiment_results.json', 'w') as f:
        json.dump({'50_epochs': results_50, '200_epochs': results_200}, f, indent=2)

    print("Done. Results saved to experiment_results.json")