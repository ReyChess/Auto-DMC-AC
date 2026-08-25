# Auto-DMC-AC v5.5.1

## Instance input visibility

- Every block in `prediction_explanations.txt` now prints the complete
  transactional item/attribute set that defines the test instance.
- `prediction_summary.csv` now includes the same input in the
  `instance_items` column.
- The explanation validator requires a nonempty input-item field for every
  prediction.

The exported values are the dataset's transactional item identifiers. Semantic
attribute/value names require the corresponding dataset-specific item mapping.
No mining, model-selection, scoring, or prediction equation was changed.
