# Strict Netconf boundary audit

## Question

The v5.4 implementation admitted positive CARs at `Netconf>=0`. Because
`Netconf=0` denotes independence between the antecedent and the class, v5.4.1
changes the positive-evidence condition to `Netconf>0`. Negative admission is
unchanged at `Netconf<0`.

## Frozen final-model audit

All 150 final models from the 15-dataset benchmark were reconstructed using
their previously selected structural variant, sigma and lambda.

- Final rules inspected: 1,116,632.
- Positive rules with exactly `Netconf=0`: 3.
- Affected final folds: Iris outer fold 2; LetRecog outer folds 2 and 3.
- Test instances with exact coverage by one of those zero rules: 114.
- Predictions changed after removing zero rules: 0/53,084.
- Correct predictions: 44,152 before and 44,152 after removal.

## Complete nested-CV rerun

The entire nested experiment was rerun with v5.4.1 strict positive admission,
using the same 15 datasets, ten outer folds, three inner folds, random seed,
sigma grid, lambda grid, support policies, computational guards and mining
budgets as the v5.4 benchmark.

Across all 150 outer folds:

- structural-variant mismatches: 0;
- selected-sigma mismatches: 0;
- selected-lambda mismatches: 0;
- outer-accuracy mismatches: 0;
- outer-macro-F1 mismatches: 0.

The resulting benchmark is therefore numerically unchanged:

- unweighted mean accuracy: 81.862664796%;
- unweighted mean macro-F1: 0.714773679860;
- pooled correct predictions: 44,152/53,084;
- pooled accuracy: 83.173837691%.

The unweighted dataset mean is the accuracy summary used in the manuscript;
the pooled value is included here only as an additional audit quantity.

## Interpretation

The inclusive boundary was a formal-definition issue, not a source of the
reported predictive results. v5.4.1 excludes statistically independent rules
from positive evidence while reproducing the complete nested benchmark.
