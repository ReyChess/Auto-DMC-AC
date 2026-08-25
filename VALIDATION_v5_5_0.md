# Validation report for v5.5.0

Date: 2026-08-26

## Build

The modified C++17 selector and the original global/CNS positive and
positive-negative miners compiled successfully with GCC/G++ on Linux. The
miners emitted only their pre-existing deprecation warning for `ftime`.

## Prediction invariance

The original v5.4.2 selector and v5.5.0 were compiled independently from the
unmodified archive and the revised source. Both were run on outer fold 1 of
Iris using global positive rules, sigma 0.10, lambda 0.15, three inner folds,
and seed 20260715. Their complete `predictions.csv` files were byte-identical:

```text
SHA-256 9e53951a960f009f6b1b9de4fc124f9715a77ce7371783df6c8343a3b270e296
```

This confirms that the explanation instrumentation did not change the tested
predictions.

## Trace verification

The ten Iris outer folds were executed with explanation export enabled. All
150 outer-test predictions were reconstructed from the exported class scores
and decision events. The validator also checked every candidate-class raw
rank-weighted score, shrinkage transformation, prior contribution, final
score, and Top-K cap. The run included 41 one-missing-item fallback predictions
and 109 exact-coverage predictions.

The distributable example under `examples/iris_fold1_explanations/` contains
15 verified prediction traces, including exact and fallback coverage.

## Scope

This validation establishes computational fidelity of the exported trace and
prediction invariance for the stated controlled run. It does not constitute a
human-subject evaluation of explanation comprehensibility, nor does it rerun
the complete 15-dataset benchmark.

The final validator additionally checks that `prediction_used_cars.csv`
contains exactly the retained rows from the complete rule trace and that all
required score, coverage, Netconf, and veto fields are populated.
