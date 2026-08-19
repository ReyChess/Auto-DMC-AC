# Auto-DMC-AC v5.4 — nested automatic support selection

This checkpoint removes the dataset-specific manual support choice from the
experimental protocol.

## Changes

1. `lambda_prior=0` is now part of the default grid:
   `0,0.05,0.10,0.15,0.20,0.30`.
2. Exact lambda ties within a structural candidate prefer the smaller lambda,
   so the no-prior model is selected when prior correction does not improve
   inner-validation macro-F1.
3. `sigma` can now be selected jointly with structural variant and lambda from
   an approximately half-decade logarithmic grid:
   `0.30,0.10,0.03,0.01,0.003,0.001`.
4. Sigma candidates are evaluated exclusively inside the outer-training data
   using the existing 3-fold inner validation.
5. Each inner mining candidate has a reproducible 30-second default mining
   budget. If `global_p` times out at a support value, all lower support values
   are rejected for that outer fold because they induce superset frequent-itemset
   search spaces.
6. The selected sigma is used for the single final mining run on the complete
   outer-training partition; the outer test partition remains untouched until
   after selection.
7. Sigma is recorded in selection, CNS-guard, PN-guard, configuration and final
   fold logs.

`--sigma VALUE` remains available as a singleton-grid compatibility option.

## Regression checks performed

- Original v5.3.2 Heart with sigma 0.001 reproduces 53.44% mean outer accuracy.
- With lambda=0 admitted but fixed sigma 0.001, Heart gives 52.13%; the change
  is explained by lambda=0 winning inner macro-F1 in two folds.
- Ionosphere with fixed sigma 0.30 reproduces 92.02% mean outer accuracy.
- Under the new resource-aware sigma grid, an Ionosphere fold evaluates
  sigma=0.30 and automatically rejects sigma=0.10 and all lower supports after
  the global positive miner reaches the 30-second budget.
- Iris with full automatic sigma/lambda selection gives 94.67% mean outer
  accuracy and 0.9463 macro-F1 in the current validation run.
- Heart with full automatic sigma/lambda selection gives 52.13% mean outer
  accuracy and 0.2334 macro-F1 in the current validation run.

