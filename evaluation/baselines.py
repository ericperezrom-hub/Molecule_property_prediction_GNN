import torch
import wandb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from utils import extract_global_features

def compute_linear_baseline(train_data, test_data, target_idx):
    X_train, valid_idx_train = extract_global_features(train_data)
    X_test, valid_idx_test = extract_global_features(test_data)

    Y_train = torch.stack(
        [train_data[i].y[0, target_idx] 
         for i in valid_idx_train])

    Y_test = torch.stack(
        [test_data[i].y[0, target_idx] 
        for i in valid_idx_test])

    model = LinearRegression()

    model.fit(
        X_train.numpy(), 
        Y_train.numpy()
    )

    Y_pred = model.predict(X_test.numpy())

    mae = mean_absolute_error(
        Y_test.numpy(), 
        Y_pred)

    return mae

def compute_tree_baseline(train_data, test_data, target_idx):
    X_train, valid_idx_train = extract_global_features(train_data)
    X_test, valid_idx_test = extract_global_features(test_data)

    Y_train = torch.stack(
        [train_data[i].y[0, target_idx] 
         for i in valid_idx_train])

    Y_test = torch.stack(
        [test_data[i].y[0, target_idx] 
        for i in valid_idx_test])

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(
        X_train.numpy(),
        Y_train.numpy()
    )

    Y_pred = model.predict(X_test.numpy())

    mae = mean_absolute_error(
        Y_test.numpy(),
        Y_pred
    )

    return mae

def compute_trivial_baseline(test_data, mean, target_idx, device):
    # Compute the trivial baseline error, 
    # which is the mean absolute error of predicting the mean value for all samples.

    # We move y_real to the device to be consistent
    # with the rest of the pipeline since mean/std
    # already live on GPU after loading the checkpoint
    # The computation itself is trivial and does not benefit from parallelization.
    y_real = test_data.y[:, target_idx].to(device)

    baseline_error = (y_real - mean).abs().mean()
    return baseline_error

def log_test_results(test_loss, baseline_error): # To be removed
    wandb.log({"test_loss": test_loss, "baseline_error": baseline_error})