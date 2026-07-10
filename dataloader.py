import torch
import torch_geometric.datasets as pyg
from torch_geometric.loader import DataLoader

data = pyg.QM9('./QM9')
n = len(data)
torch.manual_seed(0)
data = data.shuffle()

train_data = data[0:int(n*0.8)]
val_data = data[int(n*0.8):int(n*0.9)]
test_data = data[int(n*0.9):]

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

