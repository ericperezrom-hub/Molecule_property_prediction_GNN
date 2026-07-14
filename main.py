import torch
from utils import load_config
from train import get_target_stats, train_one_epoch, validate
from dataloader import get_dataloader
from models.baseline import GCN

def model_pipeline(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, test_loader, train_data = get_dataloader(config)
    mean, std = get_target_stats(train_data, config['target_idx'])

    model = GCN(num_node_features=11, hidden_dim=config['hidden_dim']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = torch.nn.L1Loss()

    best_val_loss = 1e9
    early_stopping_count = 0

    for epoch in range(config['n_epochs']):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, mean, std, config['target_idx'])
        val_loss = validate(model, val_loader, criterion, device, mean, std, config['target_idx'])
        # Early stopping checks if the model has not improved over some epochs (using the patience variable)
        # But also if the improvement has been so small that is negligible
        # If that is the case, the training is stopped
        if best_val_loss - val_loss > config['min_delta']:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            early_stopping_count = 0
        else: 
            early_stopping_count += 1

        print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")

        if early_stopping_count >= config['patience']:
            break

    test_loss = validate(model, test_loader, criterion, device, mean, std, config['target_idx'])
    print(f"Test loss: {test_loss:.4f}")

if __name__ == "__main__":
    config = load_config('config.yaml')

    if config['gpu_debug']:
        print(torch.cuda.is_available()) 
        print(torch.cuda.current_device())
        print(torch.cuda.get_device_name(0))
 
    model_pipeline(config)
