# Auto-DMC-AC — reviewer reproducibility release

This archive accompanies the manuscript **“Auto-DMC-AC: An Automatically Configured Direct Maximal Coverage Associative Classifier”** submitted to *Machine Learning with Applications*.

## Contents

* `code/`: C/C++ source for the global and class-normalized miners and the nested Auto-DMC-AC selector (version 5.5.1).
* `data/Partitions.zip`: the fixed 10-fold partitions for all 15 benchmark datasets.
* `results/main\_benchmark/`: fold-level selected-model metrics, selection logs, and feasibility-guard logs associated with the reported benchmark.
* `analysis/baselines/`: scripts and raw fold results for CART, logistic regression, Bernoulli NB, and Random Forest on the same partitions.
* `analysis/ablations/`: the confirmatory predictor-ablation summary reported in the manuscript.
* `analysis/negative\_evidence/`: the negative-conflict and directional stress-test summaries, together with the analysis scripts.
* `LICENSE`: MIT License governing reuse of the software and associated materials.
* `docs/INTERPRETABILITY\_OUTPUTS.md`: specification of the global CAR model and faithful per-prediction explanation outputs.
* `docs/USING\_YOUR\_OWN\_DATA.md`: end-to-end guide for new transactional or raw tabular datasets.
* `tools/prepare\_user\_dataset.py`: reproducible partitioning plus LUCS-KDD-style, MDLP, equal-width, and equal-frequency preprocessing.
* `tests/test\_explanation\_trace.py`: independent validator for the exported decision traces.
* `tests/user\_data/`: tests for user-data preprocessing and partition generation.
* `examples/iris\_fold1\_explanations/`: a complete verified explanation example.
* `examples/user\_dataset/toy\_dataset.csv`: minimal raw-tabular example for the preprocessing workflow.
* `VALIDATION\_v5\_5\_0.md`: build, prediction-invariance, and ten-fold trace-validation evidence.
* `VALIDATION\_USER\_DATA.md`: validation of the new user-data workflow.

## Interpretable prediction outputs

Version 5.5.1 preserves the v5.4.2 prediction logic and adds a formal
explanation-output contract. Every final outer-fold model now exports its
complete CAR base together with per-instance summaries, class-score
decompositions, the complete coverage trace, a consolidated
`prediction\_used\_cars.csv` containing only the CARs actually used, a readable
explanation, and an independent reconstruction check. These files are written under
`outer\_N/final/explanations/` when `--explanation-scope final` is active (the
default). See `docs/INTERPRETABILITY\_OUTPUTS.md` for field-level details.



## Use Auto-DMC-AC with your own data

Auto-DMC-AC can also be applied to datasets outside the supplied 15-dataset benchmark. The classifier expects integer transactional data with the class item last in each row. The repository provides `tools/prepare\_user\_dataset.py` for two reproducible entry paths:

* **already discrete / transactional data**: `transactional-cv` creates seeded stratified external partitions;
* **raw CSV / tabular data**: `outer-cv` creates fold-specific transactional representations using `lucs-kdd-style`, `mdlp`, `equal-width`, or `equal-frequency` numeric preprocessing.

For raw tabular data, preprocessing parameters are fitted on each outer-training fold and then applied unchanged to the held-out outer test. The generated directory can be passed directly to `auto\_dmc\_ac --partition-dir`. See [`docs/USING\_YOUR\_OWN\_DATA.md`](docs/USING_YOUR_OWN_DATA.md) for the complete workflow, input format, Windows commands, reproducibility metadata, and methodological notes.

The historical `code/src/DivideEn10.c` utility is retained for provenance, but the seeded Python `transactional-cv` route is recommended for new reproducible partitioning.



\### Python dependencies



The optional preprocessing toolkit for user-provided tabular datasets requires \*\*Python 3\*\* and the packages listed in \[`requirements.txt`](requirements.txt).



On Windows:



```powershell

py -3 -m pip install -r requirements.txt

```



On Linux/macOS:



```bash

python3 -m pip install -r requirements.txt

```



The Python dependencies are required \*\*only for the optional user-data preprocessing utilities\*\*. The core Auto-DMC-AC classifier and mining components remain implemented in compiled C/C++ code and do not require Python for normal execution on already prepared transactional partitions.



The preprocessing toolkit currently depends on:



\* `numpy`

\* `pandas`

\* `scikit-learn`

\* `pytest` — required for running the preprocessing validation tests



After installing the dependencies, the user-data preprocessing tests can be executed with:



\*\*Windows\*\*



```powershell

py -3 -m pytest tests/user\_data -q

```



\*\*Linux/macOS\*\*



```bash

python3 -m pytest tests/user\_data -q

```



A successful validation should report all user-data preprocessing tests as passed.



For complete instructions on preparing and running your own datasets, including already-discretized transactional data and raw tabular CSV data, see \[`docs/USING\_YOUR\_OWN\_DATA.md`](docs/USING\_YOUR\_OWN\_DATA.md).



## Main benchmark protocol

The outer evaluation uses the supplied stratified 10-fold partitions. Within every outer-training partition, three inner folds jointly select:

* minimum support: `0.30, 0.10, 0.03, 0.01, 0.003, 0.001`;
* support policy: global or class-normalized support (CNS);
* prior weight: `0, 0.05, 0.10, 0.15, 0.20, 0.30`.

Inner macro-F1 is the primary selection criterion. The outer test fold is used once, after selection. Positive-only variants form the confirmatory selector; negative evidence is analyzed post hoc as a prespecified ambiguity signal.

## Build

On Linux with CMake and a C/C++ toolchain:

```bash
cd code
./build\_linux.sh
```

The source-level usage and platform-specific build instructions are in `code/README\_ES.md` and the build scripts. Python analyses and user-data preparation require Python 3 plus `numpy`, `pandas`, `scipy`, and `scikit-learn` (`pip install -r requirements.txt`). Testing additionally requires `pytest`.

## Reference results and hardware-sensitive guards

The CSV files under `results/main\_benchmark/` are the reference outputs used in the manuscript. Candidate admission includes training-only runtime and memory guards. Consequently, low-support candidates can be rejected on slower hardware, changing the feasible candidate set and occasionally the selected configuration. This is expected behavior of the documented resource-aware protocol, not use of outer-test information.

The reference run selected global support in 113 folds and CNS in 37 folds. The reported unweighted mean performance is 81.81% accuracy and 0.7148 macro-F1.

The interpretability export does not change these reference results. It records
the already executed coverage, bounded Netconf aggregation, shrinkage, prior
adjustment, and optional gated negative-evidence decision.

## Reproducing the paper tables

* Main results: aggregate each dataset's ten rows in `results/main\_benchmark/\*/dataset\_summary.csv`.
* Baselines: run `analysis/baselines/analyze\_external\_baselines.py` after generating or using `external\_baseline\_folds.csv`; `reported\_per\_dataset\_accuracy.csv` and `reported\_statistics.csv` record the manuscript values.
* Predictor ablations: see `analysis/ablations/reported\_ablation\_summary.csv`.
* Negative evidence: see `analysis/negative\_evidence/reported\_conflict\_summary.csv` and `reported\_directional\_stress\_test.csv`.

The associated article is the authoritative description of preprocessing, guards, estimands, and statistical tests. Timing columns are descriptive and depend on hardware.

## Repository location

The public release is intended for [https://github.com/ReyChess/Auto-DMC-AC](https://github.com/ReyChess/Auto-DMC-AC).

## License

The software and associated materials are released under the MIT License. See `LICENSE` for the complete terms. Academic users are also encouraged to cite the associated article using the metadata in `CITATION.cff`.

