# Results

## Summary

| Metric | Value (eV) |
|---|---|
| Trivial Baseline MAE | 1.0674 |
| Linear regression over global features Baseline MAE | 0.7427 |
| Random forest over global features Baseline MAE | 0.4911 |
| 50-epoch Test MAE | 0.4721 ± 0.01163 |
| 200-epoch + early-stopping Test MAE | 0.4177 ± 0.01957 |
| K-fold validation (5 folds, 50 epochs) | 0.4676 ± 0.0149 |

Model: GCN, 2 layers, hidden_dim=64. See `config.yaml` for full configuration

## Baseline Comparasion

The trivial baseline predicts the mean HUMO_LUMO gap of the training set for every molecule,
ignoring the structure entirely. Our GCN trained model reduces MAE ~61% on average relative
to this baseline, confirming the model is learning meaningful structural information rather than
just fitting the target distribution.

The Linear regression baseline takes as an input global features of every molecule:
The number of carbon, hidrogen, oxigen, . and . atoms of every molecule,
The global atom and bond count, the number of rings, aromacity and the weight
and fits them into a linear model. Our GCN trained model reduces MAE ~44% on average relative
to this baseline, showing that the model captures structural and relational information 
beyond global what can be represented by global descriptors and lineal relationships.
It should be noted, however, that the molecular descriptors used manually rather than
systematically optimized. Therefore, the baseline may not represet the best possible 
performance achevable by a model based on handcrafted molecular descriptors. A potential
improvement would be to perform a more systematic feature selection.

The Random Forest baseline uses the same set of global molecular descriptors as the 
linear regression model. However, unlike linear regression, Random Forest can model 
non-linear  relationships and interactions between features by combining the predictions 
of multiple decision trees. This makes it stronger than linear regression baseline.
As we can see, our GCN trained model reduces MAE only by ~15% on average. This baseline
is also subjected to be optimized, although it is important to be aware of its limitations
since is unable to learn representations of individual atoms and their connectivity

For a fairer and more comprehensive comparison, additional baselines will be included in 
future work, such as XGBoost and Random Forest models using Morgan fingerprints 
as molecular representations.

## Cross-Validation

## Reproducibility notes

## Error analysis

## Training notes

## Change log

| Date | Change | Test MAE (eV) |
|---|---|---|
| Early Aug 2026 | Initial GCN baseline (50 epochs) | 0.4750 |
| 2026-08-10 | Extended training budget (200 epochs, early stopping) | 0.4350 |
| 2026-08-13 | Fixed non-deterministic evaluation split (seed ordering bug) | - |
| 2026-08-24 | Experiment repeated 5 times to obtain more robust results | 0.4177 ± 0.01957 |