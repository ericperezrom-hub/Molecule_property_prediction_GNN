from evaluation.baselines import compute_linear_baseline, compute_tree_baseline
from dataloader import load_split_data
from utils import load_config

if __name__ == "__main__":
    config = load_config('config.yaml')
    train_data, val_data, test_data, num_node_features = load_split_data(config)

    target_idx = config['data']['target_idx']

    linear_mae = compute_linear_baseline(train_data, test_data, target_idx)
    tree_mae = compute_tree_baseline(train_data, test_data, target_idx)

    print(f"Linear baseline MAE: {linear_mae:.4f}")
    print(f"Tree baseline MAE: {tree_mae:.4f}")