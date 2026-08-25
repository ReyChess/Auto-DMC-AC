# Auto-DMC-AC v5.5.0

## Interpretable output contract

- Added export of the complete fitted global CAR model for every outer fold.
- Added per-instance prediction summaries and per-class score decomposition.
- Added a rule-level trace distinguishing covered CARs from the Top-K CARs
  retained for the actual decision.
- Added `prediction_used_cars.csv`, a consolidated user-facing export limited
  to the positive and negative CARs actually used for each prediction, with
  class scores and veto-event fields on every row.
- Added rule rank, harmonic weight, weighted Netconf contribution, support,
  confidence inputs, exact/fallback mode, and missing fallback items.
- Added exact negative-rule trace information for gated ambiguity analysis.
- Added a concise user-readable explanation file.
- Added the complete input item/attribute set to each readable instance block
  and to `prediction_summary.csv`, linking every explanation to the actual
  test transaction.
- Added an independent prediction reconstruction check; execution fails if an
  exported trace cannot reproduce its reported prediction.
- Added `tests/test_explanation_trace.py` to verify prediction reconstruction,
  evidence caps, rank weighting, shrinkage, prior adjustment, and final scores.

No mining, selection, scoring, fallback, shrinkage, prior, or negative-veto
decision equation was changed.
