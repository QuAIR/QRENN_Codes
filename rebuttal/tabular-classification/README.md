# Tabular classification records

This archive accompanies the rebuttal comparisons on **Phishing Websites,
HTRU2 and Rice (Cammeo and Osmancik)**. It contains the records behind manuscript
Table I and the classical controls in Appendix I, Table IV.

The three examples were chosen after exploratory test-score screening.
Checkpoints and comparator hyperparameters were selected on validation data.
These results describe the selected examples and do not establish general
competitiveness or a quantum advantage.

## Contents

- `records/`: per-model metrics, development-resample means and aggregate tables
  for the three datasets. `classical-controls.csv` supplies the added appendix table.
- `frozen/`: the original scientific source snapshots. Their production-file
  hashes are checked against the model identities.
- The [data release](https://github.com/QuAIR/QRENN_Codes/releases/tag/rebuttal-tabular-2026-10-19):
  raw input archives, parsed arrays, split identities, 150 model bundles containing
  285 selected models, all candidate validation predictions, QRENN training
  histories and 570 selected-model validation/test prediction records.
- `release.json`: the download URL, size and SHA-256 checksums.
- `verification.json`: the local verification results, runtime versions,
  prediction-replay scope, rank differences and numerical warnings.

Only the three named datasets' data and experiment records are exported. Source,
model, input and prediction files keep their original bytes. The original
protocols and data manifests are retained to authenticate those bytes; the
derived `subset.json` files specify the scope of this release. Each dataset has
five development resamples and three circuit initializations per resample.
The public subset does not replace the original campaign records.

## Download and verify

Use Python 3.12 in a fresh environment, from this directory:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python reproduce.py download --out downloads
.venv/bin/python reproduce.py verify --data downloads/tabular-data
.venv/bin/python reproduce.py replay --data downloads/tabular-data
```

The download is about 135 MB. No GitHub account is needed. `verify` checks every
payload file, the original source and bundle identities, split disjointness,
all saved validation selections, all 570 prediction records and their aggregate
metrics. Raw inputs, scalers and standardized features must match exactly.
Arctangent outputs can have different byte hashes across platforms; those
differences are reported separately, and replay checks the resulting scores
against the recorded numerical tolerances. Verification performs no fitting.
`replay` also recomputes predictions from saved
parameters for resample 0 and initialization 0, covering every method and dataset
(54 validation/test records). Add `--all` to replay all 570 records, or
`--dataset rice` to restrict replay to one dataset.

Circuit replay reconstructs states with the archived NumPy implementation.
Classical replay uses saved coefficients, support rows or tree nodes, without
loading pickled estimators. The numerical tolerances are the recorded absolute
limits: `2e-8` for SVM scores and `2e-10` for the other methods, with zero
classification disagreement. Scores are never rounded to meet a tolerance.
Replay reports rank differences separately; the original saved scores remain
the authority for the published metrics.

## Training and statistical protocol

All 30, 8 and 7 released features are retained for Phishing, HTRU2 and Rice.
Preprocessing is fitted on the fitting subset. Each dataset has one fixed test
set and five fitting/validation resamples. QRENN uses two processing qubits,
eight alternating controlled input/inverse-input calls, universal SU(4)
processing blocks and a jointly trained positive-real probe. Its recorded loss
uses inverse fitting-class-frequency weights. Validation ROC-AUC selects
Phishing/Rice checkpoints; validation AP selects HTRU2 checkpoints.

The archived source contains the complete training and comparator grids.
To validate a single QRENN rerun configuration without starting training:

```sh
.venv/bin/python reproduce.py train --data downloads/tabular-data \
  --dataset rice --resample 0 --init 0 --out reruns/rice-r0-i0 --dry-run
```

Remove `--dry-run` to execute the 1,000-epoch fit in a new output directory.
Dataset choices are `phishing`, `htru2` and `rice`; resamples are 0--4 and
initializations are 0--2. This command does not replace the published models.
Phishing uses its original fixed-basis training entrypoint and archived input pin;
HTRU2 and Rice use the subsequent tabular training entrypoint.
The original runs used Linux, CPU double precision, PyTorch 2.5.1+cu124,
NumPy 2.2.6, SciPy 1.15.3 and scikit-learn 1.7.2. The requirements specify the
corresponding Python package versions; exact retraining bytes may vary across
platforms and numerical libraries.

Metrics are computed per model. Three circuit starts are averaged within each
resample, followed by the mean and sample SD across five resamples. Classical
methods contribute one validation-selected model per resample. The shared test
set means this SD describes model/resample variation, not independent test-set
replication. ROC-AUC and AP use continuous scores; accuracy uses a zero threshold,
with ties assigned to class 1. No fitting or model reselection was performed to
prepare this release.
