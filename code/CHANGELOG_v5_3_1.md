# Auto-DMC-AC v5.3.1 — lambda-grid cache optimization

## Main optimization

The inner-CV lambda grid is now evaluated with a single traversal of the validation instances and rule set per structural variant and inner fold.

For each validation instance, v5.3.1 computes once:

- exact positive coverage;
- one-missing partial fallback when exact coverage is empty;
- top-K positive evidence and shrinkage scores;
- the positive near-tie set;
- exact negative-rule coverage and negative scores for near-tie classes.

The program then evaluates every value in `--lambda-grid` using the cached, lambda-independent evidence. No rule is traversed again for the remaining lambda values.

## Compatibility

- Mining, structural variants, K=12, beta=2, exact-first coverage, partial fallback, PN veto, CNS/PN guards and output formats remain unchanged.
- The final outer model is still evaluated with the original single-lambda classifier.
- Prediction files from the optimized inner evaluator were checked against v5.3 for both P and PN variants on Iris; all `predictions_lambda_*.csv` files were identical.

## Deterministic lambda tie-break

In v5.3, exact metric ties between lambda values of the same structural variant could be broken by microsecond-level classification timing noise. Since all lambda values use the same mined rules, v5.3.1 resolves such ties first by proximity to the historical default `lambda_prior=0.15`, then by time. This makes repeated runs deterministic.

## Timing interpretation

`classification_seconds` for an inner lambda candidate is the amortized batch-classification time divided by the number of lambda values. The total batch cost is paid once per structural variant and inner fold.
