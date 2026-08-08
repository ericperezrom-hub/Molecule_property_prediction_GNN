import wandb

def log_test_results(test_loss, baseline_error):
    wandb.log({"test_loss": test_loss, "baseline_error": baseline_error})

def compute_trivial_baseline(test_data, mean, target_idx, device):
    # We move y_real to the device to be consistent
    # with the rest of the pipeline since mean/std
    # already live on GPU after loading the checkpoint
    # The computation itself is trivial and does not benefit from parallelization.
    y_real = test_data.y[:, target_idx].to(device)

    baseline_error = (y_real - mean).abs().mean()
    return baseline_error