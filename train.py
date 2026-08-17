import wandb
from utils import load_config, validate_config
from training.components import build_training_components
from training.train_utils import setup_environment, prepare_data, train_and_validate

def train_pipeline(config, weight_seed=None):
    config, device = setup_environment(config)

    run = wandb.init(project="gnn-molecule-prediction", config=config)

    train_loader, val_loader, mean, std, num_node_features = prepare_data(config, device)
    
    model, optimizer, criterion = build_training_components(config, device, num_node_features)
    
    best_val_loss, model_path = train_and_validate(
        model, train_loader, val_loader, optimizer, criterion,
        device, mean, std, config['data']['target_idx'], config, run
    )

    wandb.finish()

    return model_path, best_val_loss

if __name__ == "__main__":

    config = load_config('config.yaml')
    validate_config(config)
    train_pipeline(config, weight_seed=config['model'].get('weight_seed'))
