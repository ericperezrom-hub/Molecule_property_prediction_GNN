# Molecule_property_prediction_GNN

This project builds a graph neural network to predict molecular
properties (currently the HOMO-LUMO gap) from the QM9 dataset. The
long-term goal is an interpretable predictor, benchmarked against
classical baselines and multiple GNN architectures.

## Installation

pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

## Usage

### Train

```
python train.py
```

### Evaluate

```
python test.py --model_path <path>
```

### Demo

```
python demo.py --model_path <path> --n <n> --mode <mode>
```

### Cleanup model checkpoints

List and interactively delete checkpoints (asks for confirmation):

```
python scripts/cleanup_checkpoints.py
```

Optional flags:
- `--all`: delete all checkpoints instead of selecting indices interactively
- `--force`: skip the confirmation prompt

## Pipeline overview

The project follows a modular pipeline:

**Shared components**: `dataloader.py` (QM9 loading and splitting) and
`models/` (configurable GNN architectures) are used by both training
and evaluation.

**Training** (`training/`): training loop with early stopping, W&B logging,
   and checkpointing (model weights + normalization stats + config saved together).

**Evaluation** (`evaluation/`): loads a checkpoint and reproduces the exact
   training configuration for testing or generating a demo.

## Project structure

```
├── train.py / test.py / demo.py   # entry points
├── config.yaml                     # experiment configuration
├── dataloader.py
├── training/                       # training loop, optimizer/criterion setup
├── evaluation/                     # checkpoint loading, demo utilities
├── models/                         # model architectures + factory
├── tests/                          # unit tests
└── scripts/                        # utility scripts for project maintenance              
```
## Configuration

Experiments are configured via `config.yaml`, organized into sections:
`data`, `model`, `training`, `logging`. Example:

```yaml
model:
  model_type: "gcn"
  hidden_dim: 64
  num_layers: 2
```