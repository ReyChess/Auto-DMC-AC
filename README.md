# Auto-DMC-AC — reviewer reproducibility release

This archive accompanies the manuscript **“Auto-DMC-AC: An Automatically Configured Direct Maximal Coverage Associative Classifier”** submitted to *Machine Learning with Applications*.

## Contents

- `code/`: C/C++ source for the global and class-normalized miners and the nested Auto-DMC-AC selector (version 5.5.1).
- `data/Partitions.zip`: the fixed 10-fold partitions for all 15 benchmark datasets.
- `results/main_benchmark/`: fold-level selected-model metrics, selection logs, and feasibility-guard logs associated with the reported benchmark.
- `analysis/baselines/`: scripts and raw fold results for CART, logistic regression, Bernoulli NB, and Random Forest on the same partitions.
- `analysis/ablations/`: the confirmatory predictor-ablation summary reported in the manuscript.
- `analysis/negative_evidence/`: the negative-conflict and directional stress-test summaries, together with the analysis scripts.
- `LICENSE`: MIT License governing reuse of the software and associated materials.
- `docs/INTERPRETABILITY_OUTPUTS.md`: specification of the global CAR model and faithful per-prediction explanation outputs.
- `tests/test_explanation_trace.py`: independent validator for the exported decision traces.
- `examples/iris_fold1_explanations/`: a complete verified example containing exact and fallback predictions.
- `VALIDATION_v5_5_0.md`: build, prediction-invariance, and ten-fold trace-validation evidence.

## Interpretable prediction outputs

Version 5.5.1 preserves the v5.4.2 prediction logic and adds a formal
explanation-output contract. Every final outer-fold model now exports its
complete CAR base together with per-instance summaries, class-score
decompositions, the complete coverage trace, a consolidated
`prediction_used_cars.csv` containing only the CARs actually used, a readable
explanation, and an independent reconstruction check. These files are written under
`outer_N/final/explanations/` when `--explanation-scope final` is active (the
default). See `docs/INTERPRETABILITY_OUTPUTS.md` for field-level details.

## Main benchmark protocol

The outer evaluation uses the supplied stratified 10-fold partitions. Within every outer-training partition, three inner folds jointly select:

- minimum support: `0.30, 0.10, 0.03, 0.01, 0.003, 0.001`;
- support policy: global or class-normalized support (CNS);
- prior weight: `0, 0.05, 0.10, 0.15, 0.20, 0.30`.

Inner macro-F1 is the primary selection criterion. The outer test fold is used once, after selection. Positive-only variants form the confirmatory selector; negative evidence is analyzed post hoc as a prespecified ambiguity signal.

## Build

On Linux with CMake and a C/C++ toolchain:

```bash
cd code
./build_linux.sh
```

The source-level usage and platform-specific build instructions are in `code/README_ES.md` and the build scripts. Python analyses require Python 3 plus `numpy`, `pandas`, `scipy`, and `scikit-learn`.

## Reference results and hardware-sensitive guards

The CSV files under `results/main_benchmark/` are the reference outputs used in the manuscript. Candidate admission includes training-only runtime and memory guards. Consequently, low-support candidates can be rejected on slower hardware, changing the feasible candidate set and occasionally the selected configuration. This is expected behavior of the documented resource-aware protocol, not use of outer-test information.

The reference run selected global support in 113 folds and CNS in 37 folds. The reported unweighted mean performance is 81.81% accuracy and 0.7148 macro-F1.

The interpretability export does not change these reference results. It records
the already executed coverage, bounded Netconf aggregation, shrinkage, prior
adjustment, and optional gated negative-evidence decision.

## Reproducing the paper tables

- Main results: aggregate each dataset's ten rows in `results/main_benchmark/*/dataset_summary.csv`.
- Baselines: run `analysis/baselines/analyze_external_baselines.py` after generating or using `external_baseline_folds.csv`; `reported_per_dataset_accuracy.csv` and `reported_statistics.csv` record the manuscript values.
- Predictor ablations: see `analysis/ablations/reported_ablation_summary.csv`.
- Negative evidence: see `analysis/negative_evidence/reported_conflict_summary.csv` and `reported_directional_stress_test.csv`.

The associated article is the authoritative description of preprocessing, guards, estimands, and statistical tests. Timing columns are descriptive and depend on hardware.

## Repository location

The public release is intended for <https://github.com/ReyChess/Auto-DMC-AC>.

## License

The software and associated materials are released under the MIT License. See `LICENSE` for the complete terms. Academic users are also encouraged to cite the associated article using the metadata in `CITATION.cff`.
