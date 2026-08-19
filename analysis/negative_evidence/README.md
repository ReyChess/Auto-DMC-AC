# Negative-evidence analyses

The confirmatory selector is positive-only. The negative stage is evaluated after freezing each selected positive model.

- `reported_conflict_summary.csv` treats exact negative coverage inside a positive near tie as a risk marker.
- `reported_directional_stress_test.csv` tests the stronger policy of selecting the least-negated tied class. It reduces accuracy and is therefore not adopted by Auto-DMC-AC.
- The Python scripts document the fold-wise reconstruction and aggregation logic. Their paths are intentionally local in the archived scripts and should be set to the extracted reference results and compiled executables before rerunning.
