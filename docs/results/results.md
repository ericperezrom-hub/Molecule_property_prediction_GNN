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

The trivial baseline predicts the mean HOMO-LUMO gap of the training set for every molecule,
ignoring the structure entirely. Our 200-epoch GCN trained model reduces MAE ~61% on average relative
to this baseline, confirming the model is learning meaningful structural information rather than
just fitting the target distribution.

The Linear regression baseline takes as an input global features of every molecule:
The number of carbon, hydrogen, oxygen, nitrogen, and fluorine atoms of every molecule,
the global atom and bond count, the number of rings, aromaticity and the weight
and fits them into a linear model. Our GCN trained model reduces MAE ~44% on average relative
to this baseline, showing that the model captures structural and relational information 
beyond what can be represented by global descriptors and linear relationships.
It should be noted, however, that the molecular descriptors were used manually rather than
systematically optimized. Therefore, the baseline may not represent the best possible 
performance achievable by a model based on handcrafted molecular descriptors. A potential
improvement would be to perform a more systematic feature selection.

The Random Forest baseline uses the same set of global molecular descriptors as the 
linear regression model. However, unlike linear regression, Random Forest can model 
non-linear relationships and interactions between features by combining the predictions 
of multiple decision trees. This makes it stronger than the linear regression baseline.
As we can see, our GCN trained model reduces MAE only by ~15% on average. This baseline
is also subject to further optimization, although it is important to be aware of its limitations,
since it is unable to learn representations of individual atoms and their connectivity.

For a fairer and more comprehensive comparison, additional baselines will be included in 
future work, such as XGBoost and Random Forest models using Morgan fingerprints 
as molecular representations.

## Cross-Validation

5-fold cross-validation was run on the train+val split (90% of the
dataset), with a fixed held-out test set (10%) never used during
folds. Results were consistent across folds (std = 0.0149), indicating
the reported performance is not an artifact of a particular data split.

Note: cross-validation used a shorter training budget (50 epochs) than
the final reported model (up to 200 epochs with early stopping,
converged at epoch 122) — this was a deliberate choice to keep runtime
manageable while still validating relative consistency across folds.

## Reproducibility notes

For reproducibility purposes a two-level seeding was implemented.
Data splitting with the `split_seed` is seeded using the `set_seed()` in utils. 
For weight initialization, another independent seed, `weight_seed` is
called when building a model.

For robust comparasions across initializations, a different `weight_seed` is 
asigned to every model initialized. `split_seed` is only used when we 
want to evaluate how the model works with different data splits as in
k-fold.

It should be noted that the results are not fully deterministic even
if both seeds remain the same across initializations:
certain CUDA operations (e.g. reductions) are not strictly order-independent,
so repeated runs with identical config may show small variations.

**Known issue (fixed 2026-08-13):** `set_seed()` was previously called
in `test_pipeline`/`demo_pipeline` *after* the dataset split had already
occurred inside `load_model_from_checkpoint`, making the split
non-reproducible across evaluation runs on the same checkpoint (the
seed had no effect on a shuffle that already happened).

## Error analysis

To better understand the limitations of the GCN model, we further analyse
the distribution of prediction errors and their relationship with molecular
size, composition, chemical structure, and model initialization.

The analyses below were performed on the fixed held-out test set using five
independently trained models with the same configuration and different random
weight initializations. Each model was trained for up to 200 epochs with the
same early-stopping procedure.

### Error distribution and bias

The distribution of signed errors is approximately centred around zero for the
majority of test samples. However, a small number of outlier cases are strongly
overestimated, producing a noticeable positive tail in the error distribution.

This suggests that the model is reasonably well calibrated for typical
molecules but tends to overestimate the HOMO-LUMO gap for a small number of
particularly difficult cases.

### Error vs structural size (atoms, bonds)

The relationship between prediction error and molecular size suggests that the
model tends to produce larger errors for molecules at the extremes of the
dataset, particularly those with the lowest or highest numbers of atoms and
bonds.

The dataset is not uniformly distributed across molecular sizes: molecules with
intermediate numbers of atoms and bonds are more strongly represented. Therefore,
the increased error observed at the extremes may be partially explained by the
lower representation of these molecular sizes in the training data.

Interestingly, a slight decrease in prediction error was observed for
above-average molecular sizes compared with below-average ones. This indicates
that molecular size alone does not appear to have a simple monotonic
relationship with prediction error. Instead, the increased error at the
extremes may reflect a combination of molecular size, structural diversity, and
the distribution of training examples.

### Heteroatoms and chemical composition

The relationship between prediction error and the number of heteroatoms was
also analysed. Molecules containing between one and four heteroatoms tend to
show lower and more stable prediction errors.

In contrast, molecules with no heteroatoms or with a high number of
heteroatoms tend to exhibit larger errors. This suggests that the relationship
between molecular composition and prediction accuracy is not monotonic: adding
heteroatoms does not simply increase or decrease the prediction error.

However, the results for molecules containing six or seven heteroatoms should
be interpreted with caution, as these groups are poorly represented in the
dataset and are therefore more sensitive to individual outliers. Consequently,
the higher errors observed for these groups may not be representative of a
general limitation of the model.

Overall, these results suggest that molecular composition may influence model
performance, although the observed trends are likely to be affected by the
uneven distribution of molecular compositions in the dataset.

### Robustness across model initializations

To evaluate the robustness of the model, un addition to comparing the final
test MAE across runs, we analysed whether the models made similar prediction
errors on the same test molecules.

The signed prediction errors showed strong correlations across model
initializations, with pairwise Pearson correlation coefficients ranging from
0.90 to 0.97. The mean Pearson correlation of signed errors was 0.933, while
the mean Spearman correlation was 0.924.

Similarly, absolute prediction errors were strongly correlated across model
initializations, with Pearson correlations ranging from 0.81 to 0.91. The mean
Pearson correlation of absolute errors was 0.872, and the mean Spearman
correlation was 0.794.

These results indicate that the different model initializations tend to make
similar errors on the same molecules. In particular, molecules that are
difficult to predict for one model are generally also difficult to predict for
the other models.

#### Outlier consistency

We further analysed how consistently individual molecules were classified as
outliers across the five model initializations. Most test molecules were never
classified as outliers. However, some molecules were classified as outliers by
multiple models, with 224 molecules being identified as outliers in all five
runs.

At the same time, a substantial number of molecules were classified as outliers
by only one or a subset of the models. This indicates that not all outlier
predictions are equally robust: some difficult cases appear consistently across
all model initializations, whereas others are more sensitive to the stochastic
training process.

#### Systematic versus model-specific error

To further quantify the source of error variability, the error variance was
decomposed into a between-molecule component and a within-molecule component
across model initializations.

For signed errors, 94.4% of the total variation was explained by differences
between molecules, while only 5.6% was associated with variation across model
initializations.

For absolute errors, 89.4% of the total variation was explained by differences
between molecules, while 10.6% was associated with variation across model
initializations.

These results show that the error pattern is predominantly
molecule-dependent rather than initialization-dependent. In other words, the
main source of variation is systematic differences in how difficult different
molecular structures are for the model, rather than random variation caused by
the model initialization.

Finally, prediction difficulty was compared with disagreement across model
initializations. Molecules with larger mean absolute errors also tended to show
greater variation between model predictions. Therefore, the most difficult
molecules are not only predicted less accurately but also less consistently
across different training runs.

Overall, the robustness analysis suggests that the model is relatively stable
across random initializations. Although individual predictions can vary,
particularly for difficult molecules, the general pattern of which molecular
structures are easy or difficult to predict is highly consistent across models.

### Functional group analysis

We first compared prediction errors across broad molecular families, including
acyclic molecules, aliphatic rings, and aromatic structures. The error
distributions showed no meaningful differences between these groups, as their
medians and interquartile ranges were nearly identical.

This result is consistent with the previous analyses of heteroatom count and
molecular size, where broad structural characteristics were not sufficient to
fully explain prediction difficulty. Therefore, we extended the analysis beyond
coarse molecular categories by investigating the presence of specific
functional groups.

Approximately 85 RDKit functional group descriptors were evaluated. For each
fragment, the mean prediction error of molecules containing the fragment was
compared with that of molecules not containing it. Only fragments with a
sufficient number of samples ((n \geq 100)) were included in the analysis.

The observed differences covered a continuous range, from approximately
(0.90) to (-0.08) eV, although most functional groups were associated with
differences close to zero. This indicates that the presence of a single
functional group is generally not sufficient to explain large differences in
prediction difficulty.

The largest positive differences, corresponding to groups associated with
higher-than-average prediction errors, were observed for fragments such as
fr_Al_COO, fr_COO, and fr_COO2, with differences of approximately
(0.85-0.90) eV. However, these fragments were represented by relatively few
molecules and should therefore be interpreted with caution, as their estimated
mean errors may be strongly influenced by individual outliers.

More reliable patterns were observed for well-represented oxygen-containing
functional groups. Molecules containing fr_aldehyde ((n = 6580)) showed an
average error difference of approximately (0.27) eV, while fr_ketone
((n = 7390)) and fr_C_O ((n = 22135)) showed differences of approximately
(0.13) and (0.10) eV, respectively.

Although these effects are modest, their consistency across several
well-represented carbonyl-related descriptors suggests that molecules containing
these functional groups may be somewhat more difficult for the model to
predict.

No functional groups showed a similarly large negative association with
prediction error. Most alcohol-related descriptors, including fr_Al_OH and
fr_Al_OH_noTert, showed small negative differences, suggesting that molecules
containing these groups may be slightly easier than average to predict.

Overall, the functional group analysis suggests that coarse molecular families
such as aromaticity or ring structure do not strongly determine prediction
difficulty. Instead, specific local chemical environments may have a modest
influence on model performance. However, most observed effects are relatively
small, indicating that prediction difficulty is likely determined by a
combination of multiple structural and chemical factors rather than by the
presence of any individual functional group.

### Conclusion

Overall, the error analysis suggests that prediction difficulty is primarily
molecule-dependent rather than driven by random model initialization. Broad
structural properties such as molecular size, heteroatom count, and chemical
family show some trends but do not fully explain the observed errors.

The functional group analysis further supports this conclusion: although some
oxygen-containing and carbonyl-related groups are associated with moderately
higher errors, no single structural feature consistently determines prediction
difficulty. Overall, the results suggest that challenging cases arise from a
combination of structural and chemical factors.

## Training notes

- Early stopping with `patience` and `min_delta` (see `config.yaml`)
  was essential: an earlier run capped at 50 epochs was still improving
  when cut off. Extending the budget to
  200 epochs allowed convergence at epoch 122, improving test MAE from
  0.4750 to 0.4168.

- 200-epoch model may have a higher standard deviation when comparing different initializations
  due to the usage of early stopping.

## Change log

| Date | Change | Test MAE (eV) |
|---|---|---|
| Early Aug 2026 | Initial GCN baseline (50 epochs) | 0.4750 |
| 2026-08-10 | Extended training budget (200 epochs, early stopping) | 0.4350 |
| 2026-08-13 | Fixed non-deterministic evaluation split (seed ordering bug) | - |
| 2026-08-24 | Experiment repeated 5 times to obtain more robust results | 0.4177 ± 0.01957 |