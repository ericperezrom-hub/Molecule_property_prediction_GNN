import torch

def get_target_stats(train_data, target_idx=4):
    train_data_gaps = train_data.y[:, target_idx]
    return train_data_gaps.mean(), train_data_gaps.std()
    
def train_one_epoch(model, loader, optimizer, criterion, device, mean, std, target_idx):
    total_loss = 0
    model.train()

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch)
        target_raw = batch.y[:, target_idx]
        target = (target_raw - mean)/std
        target = target.unsqueeze(1)

        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)

def validate(model, loader, criterion, device, mean, std, target_idx):
    total_loss = 0
    model.eval()

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            out = model(batch)
            out_real = out * std + mean

            target_real = batch.y[:, target_idx].unsqueeze(1)

            loss = criterion(out_real, target_real)
            total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)