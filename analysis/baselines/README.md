# Controlled same-partition baselines

`external_baseline_folds.csv` contains fold-level outputs generated on the supplied outer partitions. `cart_complexity.csv` records the fitted tree sizes. The two `reported_*.csv` files reproduce the values in the manuscript after pairing those baseline outputs with the final Auto-DMC-AC reference run.

`run_reproducible_baselines.py` fits CART, logistic regression, Bernoulli NB, and Random Forest. `analyze_external_baselines.py` performs dataset-level Wilcoxon/Holm comparisons and the five-model Friedman analysis.
