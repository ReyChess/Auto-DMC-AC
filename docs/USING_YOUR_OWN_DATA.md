# Using your own dataset with Auto-DMC-AC

Auto-DMC-AC consumes a **discrete transactional representation**. Each non-empty line is one instance, items are integer identifiers separated by spaces, and the **last integer is the class item**:

```text
1 5 12 19 77
2 8 14 21 78
```

The repository supports two practical entry paths.

## Path A — Data already discrete / transactional

For an existing transactional file, the recommended reproducible route is:

```powershell
py -3 .\tools\prepare_user_dataset.py transactional-cv `
  --input ".\my_dataset.dat" `
  --outer-folds 10 `
  --seed 20260715 `
  --output-dir ".\MyDatasetPartition"
```

This creates stratified external folds in the layout expected by Auto-DMC-AC:

```text
MyDatasetPartition/
  Classes.dat
  Dataset1.dat
  1.dat
  ...
  Dataset10.dat
  10.dat
  preprocessing_manifest.json
```

`Classes.dat` uses the historical format:

```text
<class_item_id> <number_of_instances>
```

For example:

```text
77 664
78 731
```

The historical C utility `code/src/DivideEn10.c` is retained for provenance and compatibility with earlier workflows. For new reproducible runs, `transactional-cv` is preferred because it exposes an explicit random seed and works cross-platform.

## Path B — Raw CSV / tabular data

If the dataset contains numeric or categorical columns, use `outer-cv`. The tool creates the external train/test partitions and fits the requested preprocessing **only on each outer training partition**. The learned transformation is then applied unchanged to its held-out outer test partition.

### Dependencies

```powershell
py -3 -m pip install -r requirements.txt
```

Testing additionally requires `pytest`:

```powershell
py -3 -m pip install pytest
```

### Numeric preprocessing choices

#### `lucs-kdd-style`

Use this for historical continuity with classic associative-classification benchmarks:

```powershell
py -3 .\tools\prepare_user_dataset.py outer-cv `
  --input ".\mydata.csv" `
  --class-column class `
  --numeric-method lucs-kdd-style `
  --default-bins 5 `
  --outer-folds 10 `
  --seed 20260715 `
  --output-dir ".\MyDatasetPartition"
```

The default is five equal-width ranges. Per-column state counts can be specified explicitly:

```text
--bins-by-column "petal_length:3,petal_width:3"
```

This is deliberately called **LUCS-KDD-style**. It is not claimed to be an exact reimplementation of the historical LUCS-KDD-DN Java application, whose complete original implementation/configuration has not been recovered.

#### `mdlp`

MDLP is a supervised, class-aware discretization option for new classification problems:

```powershell
py -3 .\tools\prepare_user_dataset.py outer-cv `
  --input ".\mydata.csv" `
  --class-column class `
  --numeric-method mdlp `
  --output-dir ".\MyDatasetPartition"
```

The implementation uses the Fayyad--Irani Minimum Description Length stopping criterion. Because MDLP uses class labels, it is fitted separately on every **outer training fold**; no outer-test labels or values are used to determine its cut points. The default namespace reserves up to 256 MDLP states per numeric attribute; increase `--mdlp-max-bins` only if the program reports that the capacity was exceeded.

**Nested-selection note.** Auto-DMC-AC subsequently creates its inner folds from each already prepared outer-training representation. Therefore the outer test remains strictly held out, while the preprocessing transformation is fixed at the outer-training level rather than re-fitted again inside every Auto-DMC-AC inner split. This is consistent with treating preprocessing as the representation-construction stage for each external fold and keeps the reported outer-test evaluation independent.

#### `equal-width`

Simple unsupervised equal-width ranges:

```text
--numeric-method equal-width --default-bins 5
```

#### `equal-frequency`

Quantile-based ranges fitted from the outer training data:

```text
--numeric-method equal-frequency --default-bins 5
```

### Which method should I use?

| Goal | Suggested method |
|---|---|
| Historical continuity with LUCS-KDD-style CARM preprocessing | `lucs-kdd-style` |
| New supervised classification problem | `mdlp` |
| Simple transparent numeric baseline | `equal-width` |
| Skewed numeric distributions / approximately balanced bins | `equal-frequency` |

The preprocessing choice is part of the experimental protocol and should be reported with the results.

## Categorical variables

Each category has a stable integer item identifier. Within a given outer fold, only categories observed in the training partition are activated in the fitted mapping. An unseen test category does not expand the training-derived representation and contributes no item for that attribute.

## Missing numeric values

For each outer fold, the median is learned from the training partition and reused unchanged for the corresponding test partition.

## Stable and compact item identifiers

The preprocessing tool creates a compact global namespace for feature and class identifiers. IDs remain stable across outer folds even when MDLP or quantile binning produces a different number of active intervals. Keeping IDs compact is important because the C mining backends maintain item-indexed arrays up to the maximum item identifier.

## Reproducibility metadata

Raw-tabular preparation creates:

```text
Classes.dat
class_dictionary.csv
preprocessing_manifest.json
preprocessing_fold_1.json
...
preprocessing_fold_10.json
```

The per-fold JSON files record numeric cut points, training-fold medians, nominal mappings, item IDs, and class IDs.

## Run Auto-DMC-AC

After either preparation path, compile Auto-DMC-AC and run it against the produced partition directory. From `code/build` on Windows/Cygwin, for example:

```powershell
& "C:\cygwin64\bin\bash.exe" -lc '
./auto_dmc_ac.exe \
  --partition-dir "../../MyDatasetPartition" \
  --miner "./dmc_miner_unified.exe" \
  --output-dir "../../my_results" \
  --sigma-grid "0.001,0.003,0.01,0.03,0.10,0.30" \
  --min-netconf 0.0 \
  --lambda-grid "0.00,0.05,0.10,0.15,0.20,0.30" \
  --inner-folds 3 \
  --seed 20260715 \
  --outer-folds 10 \
  --analysis-scope final \
  --instance-log-scope final \
  --explanation-scope final
'
```

See `code/README_ES.md` for build details and the computational PN/CNS guards.

## Do not regenerate the published 15-dataset benchmark

The 15 benchmark datasets supplied with the reproducibility release are historical experimental inputs derived from the LUCS-KDD repository. Do **not** rediscretize or regenerate them when reproducing the paper. The preprocessing utilities described here are intended for **new user datasets**.

## Windows note

On Windows, `py -3` is recommended because the `python.exe` command may be intercepted by the Microsoft Store application alias:

```powershell
py -3 .\tools\prepare_user_dataset.py --help
```
