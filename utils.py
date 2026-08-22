import yaml
import subprocess
import random
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

REQUIRED_CONFIG = {
    "data": ["batch_size", "target_idx", "split_seed"],
    "model": ["model_type", "hidden_dim", "num_layers", "weight_seed"],
    "training": ["n_epochs", "lr", "patience", "min_delta", "optimizer_type", "criterion_type"],
    "logging": ["demo_project", "gpu_debug", "config_debug"],
}

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

def validate_config(config):
    missing = []

    for section, keys in REQUIRED_CONFIG.items():
        if section not in config:
            missing.append(section)
            continue

        for key in keys:
            if key not in config[section]:
                missing.append(f"{section}.{key}")

    if missing:
        raise ValueError(f"Missing config keys: {missing}")

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

def extract_global_features(data):
    X = []
    valid_indices = []
    
    for i, mol in enumerate(data):
        mol_rdkit = Chem.MolFromSmiles(mol.smiles)
        if mol_rdkit is None:
            continue   

        atom_count = mol.x[:, :5].sum(dim=0)
        num_atoms = mol.x.shape[0]
        num_bonds = mol.edge_index.shape[1] // 2
        num_rings = rdMolDescriptors.CalcNumRings(mol_rdkit)
        aromaticity = any(atom.GetIsAromatic() for atom in mol_rdkit.GetAtoms())
        molecular_weight = rdMolDescriptors.CalcExactMolWt(mol_rdkit)

        features = torch.cat([atom_count, torch.tensor([num_atoms, num_bonds, num_rings, aromaticity, molecular_weight])])
        X.append(features)
        valid_indices.append(i)

    X = torch.stack(X)
    return X, valid_indices