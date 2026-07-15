import torch
import wandb
from utils import load_config, set_seed, get_git_commit_hash
from train import get_target_stats, train_one_epoch, validate
from dataloader import get_dataloader
from models.baseline import GCN

def train_pipeline(config):
    set_seed(config['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config['git_commit'] = get_git_commit_hash()

    run = wandb.init(project="gnn-molecule-prediction", config=config)

    train_loader, val_loader, _, train_data = get_dataloader(config)
    mean, std = get_target_stats(train_data, config['target_idx'])

    model = GCN(num_node_features=11, hidden_dim=config['hidden_dim']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = torch.nn.L1Loss()

    best_val_loss = 1e9
    early_stopping_count = 0

    for epoch in range(config['n_epochs']):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, mean, std, config['target_idx'])
        val_loss = validate(model, val_loader, criterion, device, mean, std, config['target_idx'])

        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        # Early stopping checks if the model has not improved over some epochs (using the patience variable)
        # But also if the improvement has been so small that is negligible
        # If that is the case, the training is stopped
        if best_val_loss - val_loss > config['min_delta']:
            best_val_loss = val_loss
            model_path = f"best_model_{run.id}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'mean': mean,
                'std': std,
                'config': config
            }, model_path)
            early_stopping_count = 0
        else: 
            early_stopping_count += 1

        print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")

        if early_stopping_count >= config['patience']:
            break

    wandb.finish()

if __name__ == "__main__":
    config = load_config('config.yaml')

    if config['gpu_debug']:
        print(torch.cuda.is_available()) 
        print(torch.cuda.current_device())
        print(torch.cuda.get_device_name(0))
 
    train_pipeline(config)
