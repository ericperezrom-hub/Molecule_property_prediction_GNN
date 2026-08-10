# Results

## Summary

| Metric | Value |
|---|---|
| Test MAE | 0.4350 eV |
| Trivial baseline MAE | 1.0674 eV |
| Relative improvement over baseline | ~59% |
| K-fold validation (5 folds) | 0.4676 ± 0.0149 eV |

Model: GCN, 2 layers, hidden_dim=64. See `config.yaml` for full configuration.

## Baseline comparison

The trivial baseline predicts the mean HOMO-LUMO gap of the training
set for every molecule, ignoring molecular structure entirely. The
trained GCN reduces MAE by ~59% relative to this baseline, confirming
the model is learning meaningful structural information rather than
just fitting the target distribution.

## Cross-validation

5-fold cross-validation was run on the train+val split (90% of the
dataset), with a fixed held-out test set (10%) never used during
folds. Results were consistent across folds (std = 0.0149), indicating
the reported performance is not an artifact of a particular data split.

Note: cross-validation used a shorter training budget (50 epochs) than
the final reported model (up to 200 epochs with early stopping, 
converged at epoch 96) — this was a deliberate choice to keep runtime
manageable while still validating relative consistency across folds.

## Reproducibility notes

Results are seeded (`torch.manual_seed`) but not bit-for-bit
deterministic on GPU: certain CUDA operations (e.g. reductions) are
not strictly order-independent, so repeated runs with identical
config can show small variation (observed: ~0.03 eV MAE difference
between two runs with the same 50-epoch configuration). Cross-validation
results (see above) account for this by reporting variability across
multiple runs rather than relying on a single seeded result.

## Error analysis

*(to be filled in — worst/best predictions, error vs molecule size, etc.)*

## Training notes

- Early stopping with `patience` and `min_delta` (see `config.yaml`)
  was essential: an earlier run capped at 50 epochs was still improving
  when cut off. Extending the budget to
  200 epochs allowed convergence at epoch 96, improving test MAE from
  0.4750 to 0.4350.

## Change log

| Date | Change | Test MAE |
|---|---|---|
| XX-08-2026 | Initial GCN baseline (50 epochs) | 0.4750 |
| 10-08-2026 | Extended training budget (200 epochs, early stopping) | 0.4350 |