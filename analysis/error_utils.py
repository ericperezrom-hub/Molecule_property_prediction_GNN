import pandas as pd
from evaluation.checkpoint import load_model_from_checkpoint
from analysis.demo_utils import predict_all
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Fragments

def enrich_with_structure(df_run, test_data):
    num_atoms, num_bonds, num_heteroatoms = [], [], []
    for idx in df_run['index']:
        sample = test_data[idx]
        num_atoms.append(sample.z.shape[0])
        num_bonds.append(sample.edge_index.shape[1] // 2)
        num_heteroatoms.append(((sample.z != 1) & (sample.z != 6)).sum().item())

    df_run['num_atoms'] = num_atoms
    df_run['num_bonds'] = num_bonds
    df_run['num_heteroatoms'] = num_heteroatoms
    return df_run

def load_and_predict_all_checkpoints(experiment_results, split_key):
    all_dfs = []

    for run in experiment_results[split_key]:
        run['model_path'] = f"checkpoints/{run['model_path']}"
        
        bundle = load_model_from_checkpoint(run['model_path'])

        results = predict_all(
            bundle['model'], bundle['test_loader'], bundle['device'],
            bundle['mean'], bundle['std'], bundle['config']['data']['target_idx']
        )

        df_run = pd.DataFrame(results)
        df_run = enrich_with_structure(df_run, bundle['test_data'])
        df_run['run_id'] = run['weight_seed']

        all_dfs.append(df_run)

    return pd.concat(all_dfs, ignore_index=True)

def outlier_rate(group):
    q1 = group['error'].quantile(0.25)
    q3 = group['error'].quantile(0.75)
    iqr = q3 - q1
    threshold = q3 + 1.5 * iqr
    return (group['error'] > threshold).mean()

def variance_decomposition(error_matrix):

    # Convert to long format
    values = error_matrix.stack().reset_index()

    values.columns = ['molecule_id', 'run_id', 'error']

    grand_mean = values['error'].mean()

    # Mean error for each molecule
    molecule_means = (
        values
        .groupby('molecule_id')['error']
        .mean()
    )

    # Number of models per molecule
    counts = (
        values
        .groupby('molecule_id')
        .size()
    )

    # Between-molecule sum of squares
    ss_between = (
        counts *
        (molecule_means - grand_mean) ** 2
    ).sum()

    # Within-molecule sum of squares
    values = values.join(
        molecule_means.rename('molecule_mean'),
        on='molecule_id'
    )

    ss_within = (
        (values['error'] - values['molecule_mean']) ** 2
    ).sum()

    ss_total = (
        (values['error'] - grand_mean) ** 2
    ).sum()

    return {
        'total_variation': ss_total,
        'between_molecule': ss_between,
        'within_molecule': ss_within,
        'systematic_fraction': ss_between / ss_total,
        'model_specific_fraction': ss_within / ss_total
    }

def classify_family(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    num_rings = rdMolDescriptors.CalcNumRings(mol)
    is_aromatic = any(atom.GetIsAromatic() for atom in mol.GetAtoms())

    if is_aromatic:
        return 'aromatic'
    elif num_rings > 0:
        return 'aliphatic_ring'
    else:
        return 'acyclic'
    
def add_chemical_family(df, test_data):
    families = []
    for idx in df['index'].unique():
        sample = test_data[idx]
        family = classify_family(sample.smiles)
        families.append({'index': idx, 'family': family})

    family_df = pd.DataFrame(families).drop_duplicates(subset='index')
    return df.merge(family_df, on='index', how='left')

def add_fragment_features(df, test_data):
    fragment_fns = [name for name in dir(Fragments) if name.startswith('fr_')]

    rows = []
    for idx in df['index'].unique():
        sample = test_data[idx]
        mol = Chem.MolFromSmiles(sample.smiles)
        if mol is None:
            continue
        row = {'index': idx}
        for name in fragment_fns:
            row[name] = getattr(Fragments, name)(mol)
        rows.append(row)

    fragment_df = pd.DataFrame(rows)
    return df.merge(fragment_df, on='index', how='left')

def analyze_fragments(df, fragment_fns, min_count=100):
    results = []
    for name in fragment_fns:
        has_fragment = df[name] > 0
        n_with = has_fragment.sum()
        n_without = (~has_fragment).sum()

        if n_with < min_count or n_without < min_count:
            continue 

        mean_with = df.loc[has_fragment, 'error'].mean()
        mean_without = df.loc[~has_fragment, 'error'].mean()

        results.append({
            'fragment': name,
            'n_with': n_with,
            'mean_error_with': mean_with,
            'mean_error_without': mean_without,
            'diff': mean_with - mean_without
        })

    return pd.DataFrame(results).sort_values('diff', ascending=False)