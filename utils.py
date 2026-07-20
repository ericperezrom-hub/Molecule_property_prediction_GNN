import yaml
import subprocess
import random
import torch
import numpy as np

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def print_config(config):
    print("\nUSED CONFIGURATION:")
    for section, values in config.items():
        if isinstance(values, dict):
            print(f"{section}:")
            for k, v in values.items():
                print(f"  {k}: {v}")
        else:
            print(f"{section}: {values}")

def print_gpu_stats():
    print("\nGPU STATS:")
    available = torch.cuda.is_available()
    print(f"CUDA available: {available}")
    if available:
        print(torch.cuda.current_device())
        print(torch.cuda.get_device_name(0))
    
def get_git_commit_hash():
    # Saves which version of the code ran a experiment
    # A experiment can be run in that version using `git checkout <hash>`
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).decode('ascii').strip()
    except Exception:
        return "unknown"
    
def set_seed(seed):
    # Sets seed for shuffling datasets and other aplications
    # Note: The experiments are not fully deterministic due to non-deterministic
    # GPU operationns
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)