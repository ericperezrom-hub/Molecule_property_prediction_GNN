import yaml
import subprocess
import random
import torch
import numpy as np

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
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