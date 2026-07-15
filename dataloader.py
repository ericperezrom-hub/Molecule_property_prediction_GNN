import torch
import torch_geometric.datasets as pyg
from torch_geometric.loader import DataLoader

def get_dataloader(config):
    dataset = pyg.QM9('./QM9')
    n = len(dataset)

    torch.manual_seed(config['seed'])
    dataset = dataset.shuffle()

    train_data = dataset[0:int(n*0.8)]
    val_data = dataset[int(n*0.8):int(n*0.9)]
    test_data = dataset[int(n*0.9):]

    batch_size = config['batch_size']

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_data