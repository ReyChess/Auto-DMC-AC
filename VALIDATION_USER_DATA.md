# User-data workflow validation

The repository now includes a complete path for applying Auto-DMC-AC to datasets outside the fixed 15-dataset benchmark.

## Supported inputs

1. **Already-discrete transactional data**: `tools/prepare_user_dataset.py transactional-cv` creates seeded stratified external partitions.
2. **Raw CSV/tabular data**: `tools/prepare_user_dataset.py outer-cv` supports `lucs-kdd-style`, `mdlp`, `equal-width`, and `equal-frequency` numeric preprocessing.

The class item is always the last integer in each generated transaction.

## Leakage boundary

For raw tabular data, fitted cut points, medians, and active nominal mappings are learned from each **outer training partition** and applied unchanged to the corresponding outer test. The outer test is never used to fit those transformations. Auto-DMC-AC then performs its normal inner selection on the already prepared outer-training representation; see `docs/USING_YOUR_OWN_DATA.md` for the exact methodological scope.

## Compact item namespace

Generated item IDs are stable across outer folds and deliberately compact. This matters because the C miners allocate item-indexed structures up to the maximum item ID. Earlier development prototypes using very large sparse IDs were rejected before release.

## Local automated validation

The integrated Python test suite completed successfully:

```text
9 passed
```

It covers MDLP splitting, equal-width/equal-frequency state counts, compact/stable class IDs, raw-tabular CLI generation, and transactional CLI generation.

## End-to-end smoke validation

A synthetic two-class raw CSV with numeric and categorical attributes was processed with MDLP into three external folds and then passed directly to the compiled Auto-DMC-AC executable. The full run completed successfully (`status 0`), produced all three outer-fold summaries, and generated explanation/trace-verification output for every outer fold.

## LUCS-KDD terminology

`lucs-kdd-style` is a compatibility-oriented preprocessing option using the historically common five equal-width ranges by default, with explicit per-column overrides. It is **not** presented as an exact reimplementation of the unavailable historical LUCS-KDD-DN Java application. The published 15-dataset benchmark remains unchanged and should not be regenerated with this utility.
