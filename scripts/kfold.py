import argparse
from utils import load_config, validate_config
from evaluation.kfold import run_kfold

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=5, help="Number of folds")
    args = parser.parse_args()

    config = load_config('config.yaml')
    validate_config(config)

    mean_loss, std_loss = run_kfold(config, k=args.k)

    print(f"K-Fold results ({args.k} folds):")
    print(f"Mean val loss: {mean_loss:.4f}")
    print(f"Std val loss: {std_loss:.4f}")
