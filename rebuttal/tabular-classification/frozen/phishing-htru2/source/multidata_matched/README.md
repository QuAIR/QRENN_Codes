# Four-dataset matched-circuit exploratory benchmark

Only this package extends the frozen prior studies. Official archives are staged
offline; no feature selection, duplicate removal or held-out model selection is
allowed. This is an exploratory benchmark after earlier dataset/test exploration.

Preparation: `python -m multidata_matched.data --raw RAW --out DATA --legacy LEGACY_DATA`.
This authenticates official archive hashes, preserves original row order/labels,
and enforces the legacy Phishing/Spambase split and fit-only scaler equality.
`load_split(DATA, dataset, resample)` returns fitting and validation sections only;
held-out loading requires an explicit `include_test=True` evaluation call.

`python -m multidata_matched.train --data DATA --create-pin PIN` freezes the source
closure, literal protocol and data manifest. The closure follows repository-local
imports from every non-test module and includes imported package protocols. Any
later package modules require a fresh final pin before formal jobs. Formal jobs
use `--data DATA --out RUN --pin PIN --dataset magic --r 0 --i 0`; `--resume`
authenticates progress and an immutable checkpoint generation embedding the selected parameters, history, optimizer, scheduler and shuffle. Assessment never overwrites artifacts used by the published generation. Unpublished generations survive as evidence but cannot drive resume; best/history exports are materialized at completion. After publication, only the current and previous successful checkpoint generations remain live. Detected unpublished generations are archived once on authenticated resume and retained as interrupted evidence.
The grid has exactly 45 new runs (MAGIC, HTRU2, Spambase). HTRU2 checkpoint
selection uses validation AP; the others use validation ROC-AUC, with loss/epoch
ties. Threshold is always zero. Training has 1000 epochs and no refit/early stop.

`phishing_source(new_data, legacy_data, qrenn_root, r, i)` authenticates an existing
sealed fixed-basis run and its old dependency pin, compares all new/old fitting
and validation arrays/scalers, and replays selected validation predictions. It
returns source identities, initial/best/terminal parameters and validation scores;
it never modifies or retrains the old run. Its source record is suitable for
binding later kernel-adapter provenance. Original legacy artifacts stay external.

`fit_synthetic` is a separate maximum-30-epoch, maximum-128-row fixture API that refuses official hash identities whose completions cannot
satisfy formal authentication. Tests run only bounded synthetic models; official
data tests prepare/authenticate and replay a sealed source without fitting.

Development comparison and evaluation are separate controller operations. After
final source review, prepare the data again and create a fresh training pin: the
manifest binds the now complete protocol, including every declared comparison
grid. Training output layout is `QRENN_ROOT/results/{dataset}/r{r}/i{i}`.

```
python -m multidata_matched.campaign create --data DATA --pin PIN --out COMPARISONS \
  --qrenn-root QRENN_ROOT --phishing-kernel-root OLD/campaign \
  --legacy-data LEGACY_DATA --legacy-qrenn OLD/dependencies/qrenn
python -m multidata_matched.campaign run-development --out COMPARISONS --workers 8
# Explicit controller instruction, only after all 200 bundles are authenticated:
python -m multidata_matched.campaign freeze --out COMPARISONS --snapshot FREEZE.json
# Separate explicit controller instruction; this alone enables held-out access:
python -m multidata_matched.campaign evaluate --data DATA --snapshot FREEZE.json --out EVALUATION
```

`OLD/campaign` retains its `pin.json` and `results/{initial,trained}/r{r}/i{i}`.
External path strings in its sealed records are provenance, not relocation
requirements. Imported Phishing kernels authenticate old source/completion/pin,
all fitting/validation row identities, scaler and paired circuit parameters;
they never refit. Completed imported bundles contain portable copies of the
source identities/pin, parameters, all candidate validation predictions and
selected original support rows. No Gram matrix or Python pickle is retained.

The campaign has exactly 200 bundles: 60 QRENN imports, 120 dual-kernel bundles,
and 20 four-comparator bundles, totaling 380 selected models. Its durable
`campaign.json` records each attempt in `attempts/{task-name}/{unique-id}`.
At most eight comparison processes run concurrently, each with one numerical
thread. Repeating `run-development` authenticates completed work and gives each
failed/interrupted task a fresh attempt directory. Use `--limit N` for a bounded
controller batch. After a controller crash, `--recover-stale-lock` removes its
lock only when the saved controller PID no longer exists; it never interrupts a
live process. No development operation opens the test section. Explicit
freeze rejects any missing coordinate, changed source/data, incomplete method,
wrong formal grid, candidate-metric mismatch or changed selected prediction.

Portable inference: `kernels.predict_original_features(BUNDLE, raw_x, method)`
and `baselines.predict_original_features(BUNDLE, raw_x, method)`. Kernel methods
are `fidelity` and `projected`; QRENN uses `qrenn_zz`. Classical methods are
`lr_standardized`, `lr_angles`, `rbf_svm`, and `gradient_boosting`. Linear models
save coefficients, RBF models save raw support rows and dual coefficients, and
boosting saves numeric tree nodes and its raw-score baseline. Every recipe uses
threshold zero and records no refit. Initial kernel scores are separately named
and do not become QRENN trained accuracy.

Evaluation writes immutable per-model validation/test NPZ predictions,
`runs.csv`, `resample_means.csv`, `summary.json`, `table.csv`,
`paired_differences.csv`, `replay_audit.json`, and `independent_audit.json`.
The last file records a separate post-output check of every candidate selection,
every saved prediction metric and independently calculated hierarchy arithmetic. Every reported table metric is
recomputed from saved predictions. Hierarchy averages three starts per resample
before averaging five resamples with sample SD. The independent replay constructs
original Gell-Mann matrices and full circuit states with NumPy/SciPy; it never
calls the production feature-state function. It reports exact score errors,
classification disagreements and ranking disagreements without rounding. Tiny
rank changes due to floating point ties remain visible in the report.

Bounded tests: `python -m pytest multidata_matched -p no:cacheprovider --basetemp OUTPUT/tests/RUN`.
Set `QRENN_EVIDENCE_ROOT` to a portable copy of the evidence output tree for the
official archive/source integration tests. Missing evidence fails explicitly;
scientific requirements are never skipped. The synthetic APIs limit input and
circuit size, use a separate evidence mode, and cannot satisfy a formal freeze.
