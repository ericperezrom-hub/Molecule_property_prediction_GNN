import torch
import wandb
from utils import set_seed, get_git_commit_hash, print_config, print_gpu_stats
from training.loop import get_target_stats, train_one_epoch, validate
from dataloader import get_dataloader

def setup_environment(config):
    if config['config_debug']:
        print_config(config)

    set_seed(config['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if config['gpu_debug']:
        print_gpu_stats()

    config['git_commit'] = get_git_commit_hash()

    return device
    
def prepare_data(config, device):
    train_loader, val_loader, _, train_data, _, num_node_features = get_dataloader(config)

    mean, std = get_target_stats(train_data, config["target_idx"])

    return (
        train_loader,
        val_loader,
        mean.to(device),
        std.to(device),
        num_node_features,
    )

def train_and_validate(model, train_loader, val_loader, optimizer, criterion,
                        device, mean, std, target_idx, config, run):
    best_val_loss = 1e9
    early_stopping_count = 0

    for epoch in range(config['n_epochs']):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                      device, mean, std, target_idx)
        val_loss = validate(model, val_loader, criterion, device, mean, std, target_idx)

        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

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

    return best_val_loss
