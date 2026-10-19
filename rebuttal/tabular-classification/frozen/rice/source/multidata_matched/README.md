# Online Shoppers and Rice matched-circuit extension

This package runs the two declared datasets after the original four-dataset
exploration. Prior numeric dataset and legacy-import paths remain available for
authentication tests; the new campaign does not invoke those import branches.
All old model, loss, optimizer and inference implementations remain unchanged.
This is an exploratory benchmark after earlier dataset/test exploration.

Prepare staged official archives offline:

```
python -m multidata_matched.data --raw RAW --out DATA
python -m multidata_matched.train --data DATA --create-pin PIN
```

Preparation uses exactly `protocol.json`'s dataset list and rejects duplicate or
unknown names. `--legacy LEGACY_DATA` is optional and explicitly requests the old
Phishing/Spambase equality check when preparing those historical datasets.
`load_split(DATA, dataset, resample)` exposes only fit/validation sections.
Held-out rows require explicit evaluation access.

Online Shoppers retains all 17 raw columns: first ten numeric, last seven
nominal. Parsed NPZ stores exact raw tokens, original Revenue labels, reversible
token IDs, feature names and the complete token dictionary. The dictionary is
also in the manifest. The IDs are parsing labels, never model coordinates.
Each split saves a vocabulary learned from fit rows alone. Validation/test
unknown categories produce zero for that nominal attribute's indicators.
The model-facing `x` is the unscaled numeric-plus-one-hot matrix; StandardScaler
is fit on transformed fit rows only. All coordinates use `0.5+atan(z)/pi`.
Dimensions are 74,73,75,73,73 and n=7. Rice retains all seven released numeric
attributes, original Cammeo/Osmancik labels and lexical tokens; n=3.
`data.audit_reconstruction(DATA, dataset, resample)` separately reconstructs all
encoded columns from raw tokens and manually checks fit-only statistics, rather
than using the transform helper. Parsed and split arrays and official archives
are authenticated through manifest and split hashes.

The literal training grid has 30 fits: two datasets, five resamples, three starts.
Initialization is 82000+100*r+i; shuffle adds 1000000. Formal jobs use
`--data DATA --out RUN --pin PIN --dataset shoppers --r 0 --i 0` (or rice).
Every job reaches 1000 epochs with unchanged selection and no refit/early stop.
Selection uses validation AP for Shoppers and ROC-AUC for Rice, with unchanged
loss/epoch ties. The threshold is zero. `--resume` authenticates checkpoint,
optimizer, scheduler, history and shuffle state. Failed attempts remain evidence.
Synthetic APIs retain their original bounds and cannot satisfy formal evidence.

After source review, prepare again and pin the complete final source closure
before any fitting. Training layout is `QRENN_ROOT/results/{dataset}/r{r}/i{i}`.
Development comparison and held-out evaluation are separate controller actions:

```
python -m multidata_matched.campaign create --data DATA --pin PIN --out COMPARISONS \
  --qrenn-root QRENN_ROOT --phishing-kernel-root OLD/campaign \
  --legacy-data LEGACY_DATA --legacy-qrenn OLD/dependencies/qrenn
python -m multidata_matched.campaign run-development --out COMPARISONS --workers 8
python -m multidata_matched.campaign freeze --out COMPARISONS --snapshot FREEZE.json
```

The historical path arguments remain accepted for compatibility and are unused
for these two datasets. Exactly 100 bundles contain 30 QRENN, 120 selected kernel
and 40 classical models (190 selected models). Freeze rejects missing/duplicate
coordinates or selected methods, changed source/data, wrong grids or candidate
selection mismatches. Literal counts are independently checked against the
expected coordinate/method inventory. Freeze must be explicitly requested after
all bundles complete; development never automatically enables held-out access.

Use `evaluation_parallel.evaluate(DATA, FREEZE, EVALUATION, workers=8)` for the
new campaign's explicit evaluation. It authenticates the supplied new freeze and
`DATA/../pin.json` before held-out loading. It uses at most eight spawned workers
and one numerical thread per worker, isolated worker outputs, no retries, frozen
collection order, complete inventories and parent-only atomic completion sealing.
The source closure includes the evaluator and its numerical helper.

The new numerical policy was declared before fitting: absolute tolerance 2e-8
for SVM scores, 2e-10 for other scores, relative tolerance zero, and no
classification disagreement. The 2e-10 strict reference remains reported for
all scores. Exact predictions, affected ranks and all four metric differences
are retained without rounding. Earlier four-dataset roundoff investigation is
recorded only as numerical precedent; its source/snapshot cannot authorize this
campaign. The old `evaluate` serial API remains a strict reference for bounded
regression comparisons; formal delivery uses `evaluation_parallel`.

Portable saved-model inference contracts remain unchanged. For Shoppers,
`predict_original_features` takes the authenticated unscaled transformed `x`;
use the separately saved split vocabulary to reconstruct it from raw attributes.
QRENN uses original ZZ readout. Initial/trained fidelity/projected SVMs remain
separate classifiers. Logistic, RBF and boosting models save portable parameters,
support rows or numeric trees without Python pickles. No evaluation fits a model.

Evaluation produces 380 validation/test prediction records, per-run metrics,
resample/start summaries, paired differences and replay audits. Quantum replay
uses independent NumPy/SciPy full states and original Gell-Mann matrices.
Classical portable replay shares its implementation and is explicitly labeled
as such. Independent post-output auditing checks candidates, saved metrics and
hierarchy arithmetic: average three starts within resample, then mean/sample SD
over five resamples sharing one fixed test population.

Run bounded tests with `python -m pytest multidata_matched -p no:cacheprovider
--basetemp OUTPUT/runtime/tests/UNIQUE_RUN`. `QRENN_EVIDENCE_ROOT` can point to a
portable copy of earlier official inputs/evidence for legacy authentication
fixtures. Missing evidence fails explicitly; those fixtures do not refit legacy
models. Official new data tests only parse, prepare, authenticate and reconstruct.
