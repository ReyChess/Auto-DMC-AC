# Interpretable outputs

Auto-DMC-AC v5.5.1 exports the fitted direct-CAR model and a faithful trace for
every outer-test prediction by default. The predictive equations and selected
class are unchanged; the new files expose the computational objects already
used by the classifier.

Each `outer_N/final/explanations/` directory contains:

- `global_rule_model.csv`: every persistent positive or negative CAR in the
  fitted outer-fold model, including antecedent, consequent, support,
  confidence, Netconf, and a stable source-line rule identifier.
- `prediction_summary.csv`: one row per test instance with prediction,
  the complete transactional input item list, coverage mode, rule counts,
  ambiguity and negative-evidence events, and the independently reconstructed
  prediction.
- `prediction_class_scores.csv`: one row per instance and class, exposing the
  rank-weighted score before shrinkage, the common shrinkage target, stabilized
  score, prior contribution, and final comparison score.
- `prediction_rule_trace.csv`: one row per covered CAR. The
  `retained_for_decision` field distinguishes the rules actually used by the
  bounded Top-K computation from additional covered rules. Fallback rows also
  identify the missing antecedent item. Positive rows expose their harmonic
  rank weights and additive numerators. Negative aggregation is non-additive;
  retained negative rows identify the exact counter-evidence consulted, while
  the resulting negative score is reported at class level.
- `prediction_used_cars.csv`: the user-facing decision table containing only
  CARs actually retained by the bounded positive aggregation or consulted by
  the exact negative stage. Every row includes antecedent, consequent class,
  Netconf, coverage type, retained rank, class scores, and the complete
  negative-evidence/veto state for that prediction.
- `prediction_explanations.txt`: a compact human-readable explanation for each
  test prediction. Each block begins with the complete set of input
  items/attributes that defines the instance.
- `trace_verification.csv`: prediction-by-prediction verification that the
  exported final scores and any gated veto reconstruct the reported label.

The feature and class identifiers are the identifiers used by the transactional
dataset. Their domain meaning depends on the dataset's item mapping. The export
therefore establishes computational fidelity and inspectability, not automatic
human comprehension or causal validity.

## Validation

Run the supplied validator on any explanation directory:

```bash
python3 tests/test_explanation_trace.py \
  auto_dmc_results/outer_1/final/explanations
```

The validator checks all reported predictions, the positive and negative caps,
rank weights, raw score, shrinkage equation, prior contribution, and final
score. It exits with a nonzero status if any invariant fails.

## Disabling the export

The default is `--explanation-scope final`. For a performance-only run, use:

```text
--explanation-scope none
```

This flag changes only the persistence of explanations, not prediction logic.
