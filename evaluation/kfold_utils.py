import wandb
import torch
import torch_geometric.datasets as pyg
from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader
from training.components import build_training_components
from training.train_utils import train_and_validate
from training.loop import get_target_stats
from utils import get_git_commit_hash

def load_split_data_kfold(config, k):
    dataset = pyg.QM9('./QM9')
    dataset = dataset.shuffle()

    if config['data']['data_debug']:
        dataset = dataset[:config['data']['size_debug']]
    
    n = len(dataset)

    # Mantain the last 10% of the dataset as the test set,
    # and use the first 90% for k-fold cross-validation.
    test_data = dataset[int(n * 0.9):]           
    trainval_data = dataset[:int(n * 0.9)]         

    fold_size = len(trainval_data) // k
    folds = [trainval_data[i*fold_size : (i+1)*fold_size] for i in range(k)]

    num_node_features = dataset.num_node_features

    return folds, test_data, num_node_features

def build_fold(folds, fold_index):
    val_data = folds[fold_index]

    train_folds = [folds[j] for j in range(len(folds)) if j != fold_index]
    train_data = ConcatDataset(train_folds)

    return train_data, val_data

def run_kfold(config, k=5):
    folds, test_data, num_node_features = load_split_data_kfold(config, k)

    val_losses = []

    for i in range(k):
        run_id = get_git_commit_hash()

        train_data, val_data = build_fold(folds, i)
        train_loader = DataLoader(train_data, batch_size=config['data']['batch_size'], shuffle=True)
        val_loader = DataLoader(val_data, batch_size=config['data']['batch_size'], shuffle=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mean, std = get_target_stats(train_data, config['data']['target_idx'])
        model, optimizer, criterion = build_training_components(config, device, num_node_features)

        run = wandb.init(project="gnn-molecule-prediction-kfold", config=config, group=f"kfold_{run_id}")

        best_val_loss = train_and_validate(model, train_loader, val_loader, optimizer, criterion, device, mean, std, config['data']['target_idx'], config, run)

        wandb.finish()

        val_losses.append(best_val_loss)

    val_losses_tensor = torch.tensor(val_losses)

    return val_losses_tensor.mean().item(), val_losses_tensor.std().item()