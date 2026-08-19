# Prior-scope diagnostic for v5.4.2

Before changing the implementation, the v5.4.1 decision was reconstructed on
the 150 archived outer folds. The reconstruction reproduced all archived
outer-fold accuracy and macro-F1 values exactly.

Keeping each fold's already selected support policy, sigma and lambda fixed,
the formal global prior-adjusted decision was then compared with the v5.4.1
hybrid tie-restricted behavior over all 53,084 out-of-fold predictions.

- Changed predictions: 7 / 53,084 (0.013187%).
- v5.4.1 correct predictions: 44,152.
- Global-prior correct predictions: 44,158.
- Wrong -> correct: 6.
- Correct -> wrong: 0.
- Wrong -> different wrong class: 1.
- Unweighted mean dataset accuracy: 0.818626647964 -> 0.819088678340.
- Unweighted mean dataset macro-F1: 0.714773679860 -> 0.714907209837.

Only Flare (5 changed predictions, all corrected) and Heart (2 changed
predictions, one corrected and one still incorrect) were affected in this
frozen-configuration diagnostic.

These numbers are a regression/consistency diagnostic, not replacement nested
cross-validation results. Because v5.4.2 also changes the decision used during
inner validation, official benchmark results must come from a fresh complete
nested run if they are to be attributed to v5.4.2.
