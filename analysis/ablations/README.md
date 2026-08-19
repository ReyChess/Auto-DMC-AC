# Predictor ablations

`reported_ablation_summary.csv` records the table reported in the manuscript. The interventions reuse each fold's selected positive rule base and change only the prediction mechanism: disable partial fallback, set shrinkage to zero, or retain one covering rule per class (`K=1`). Partial fallback is used 910 times among 53,084 out-of-fold predictions.
