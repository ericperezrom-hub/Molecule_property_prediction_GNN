import torch_geometric.datasets as pyg
from torch_geometric.loader import DataLoader

def load_split_data(config):
    dataset = pyg.QM9('./QM9')
    n = len(dataset)
    dataset = dataset.shuffle()

    train_data = dataset[0:int(n*0.8)]
    val_data = dataset[int(n*0.8):int(n*0.9)]
    test_data = dataset[int(n*0.9):]

    num_node_features = dataset.num_node_features

    return train_data, val_data, test_data, num_node_features

def get_dataloader(config):
    train_data, val_data, test_data, num_node_features = load_split_data(config)

    batch_size = config['data']['batch_size']
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_data, test_data, num_node_features