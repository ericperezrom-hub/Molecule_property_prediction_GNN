# Molecule Property Prediction with Graph Neural Networks

## Overview 

This project implements a Graph Neural Networks (GNN) to predict the
HOMO-LUMO energy gap of molecules from the QM9 dataset.

Molecules are represented as graphs, where atoms correspond to nodes and bonds
correspond to edges. The project includes training, evaluation, cross-validation,
classical machine learning baselines, and an analysis of prediction errors.

The current implementation focuses on a GCN and compares its performance
against trivial, linear regression, and random forest baselines based on global
molecular descriptors.

## Results

The current best model is a 2-layer Graph Convolutional Network (GCN) with a
hidden dimension of 64, trained for up to 200 epochs with early stopping.

The final result, evaluated across five independent model initializations, is:

| Metric | Result |
|---|---:|
| Test MAE | 0.4177 ± 0.0196 eV |
| 5-fold CV MAE (50 epochs) | 0.4676 ± 0.0149 eV |
| Trivial baseline MAE | 1.0674 eV |
| Linear regression baseline MAE | 0.7427 eV |
| Random forest baseline MAE | 0.4911 eV |

The GCN substantially outperforms the trivial and linear baselines and achieves
a moderate improvement over the random forest baseline based on global
molecular descriptors.

Prediction errors were analysed across multiple dimensions, including molecular
size, heteroatom count, chemical families, functional groups, and independent
model initializations.

The analysis suggests that prediction difficulty is primarily
molecule-dependent rather than driven by random model initialization. Broad
structural descriptors alone do not fully explain prediction difficulty, while
some oxygen-containing and carbonyl-related functional groups show a modest
association with higher prediction errors.

See [`results.md`](results.md) for the complete analysis.

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd Molecule_property_prediction_GNN
```

Install the project dependencies:

pip install -r requirements.txt

## Usage

### Train a model

```
python scrpits/train.py
```

Loads data and uses it to train a brand new model.
This instruction generate a `.pt` checkpoint which can be evaluated later.

### Train multiple models

```
python scripts/run_experiments.py
```

### Evaluate a model

```
python scripts/test.py --model_path <path>
```

### Build a demo for a model

```
python scripts/demo.py --model_path <path> --n <n> --mode <mode>
```

### Cross-validation (used for checking that models are similar when using different data splits)

```
python scripts/kfold.py --k <n>
```

### Run classical baselines and print results

```
python scripts/classical_baselines.py --k <n>
```

### Run unit tests

```
pytest tests/
```

### Cleanup model checkpoints

List and interactively delete checkpoints (asks for confirmation):

```
python scripts/cleanup_checkpoints.py
```

Optional flags:
- `--all`: delete all checkpoints instead of selecting indices interactively
- `--force`: skip the confirmation prompt

## Configuration

Experiments are configured via `config.yaml`, organized into sections:
`data`, `model`, `training`, `logging`. 

## Project structure


├── requirements.txt             # Project dependencies
├── utils.py                     # General utilites for seedingm gut tracking
├── dataloader.py                # Dataset loading and splitting
├── config.yaml                  # Experiment configuration
├── analysis/                    # Utilities for error analysis abnd demos
├── docs/                        # Project documentation and detailed result
├── evaluation/                  # Checkpoint loading, baseline comparasion, k-fold logic
├── models/                      # Model architectures + factory
├── notebooks/                   # Exploratory analysis and experiment notebooks
├── scripts/                     # Executable pipelines
├── tests/                       # Unit tests
└── training/                    # Training loop, optimizer/criterion setup           
```

## Future work

- Implement edge features and global features into a GNN model
- Try other types of GNNs, such as GAT, GIN or Transformers
- Implement hyperparameter search
- Make a interpretability analysis