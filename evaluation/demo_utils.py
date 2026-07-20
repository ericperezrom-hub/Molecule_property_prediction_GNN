import torch
import random
import wandb
from rdkit import Chem
from rdkit.Chem import Draw

def predict_all(model, loader, device, mean, std, target_idx):
    model.eval()

    idx = 0
    results = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            out = model(batch)
            out_real = out * std + mean

            target_real = batch.y[:, target_idx].unsqueeze(1)

            for i in range(batch.num_graphs):
                pred = out_real[i].item()
                gt = target_real[i].item()

                error = abs(pred - gt)

                results.append({
                    'index': idx,
                    'prediction': pred,
                    'ground_truth': gt,
                    'error': error
                })
                idx += 1
    
    return results

def select_examples(results, n, mode="random"):
    if mode == "random":
        return random.sample(results, n)

    elif mode == "worst":
        return sorted(results, key=lambda x: x['error'], reverse=True)[:n]

    elif mode == "best":
        return sorted(results, key=lambda x: x['error'])[:n]
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'random', 'worst' or 'best'.")
    
def mol_to_image(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"SMILES could not be parsed: {smiles}")
    rep = Draw.MolToImage(mol)
    return rep
    
def log_demo_table(examples, test_data, config):
    wandb.init(project= config['demo_project'], config=config)

    table = wandb.Table(columns = ["molecule", "smiles", "prediction", 
                                   "ground_truth", "error"])
    
    for example in examples:
        sample = test_data[example['index']]

        try:
            image = mol_to_image(sample.smiles)
        except ValueError as e:
            print(f"Skipping molecule: {e}")
            continue

        table.add_data(
            wandb.Image(image),
            sample.smiles,
            example['prediction'],
            example['ground_truth'],
            example['error']
        )

    wandb.log({ "demo_examples": table })

    wandb.finish()