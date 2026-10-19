# Current manuscript experiment records

This release archives existing inputs, seeds, source, model parameters, training
histories, predictions and measurement records for the revised manuscript.
The manuscript snapshot is `a344097897f36291a7a46fd18f462a293a874443`.
No training, inference, gradient calculation or statistical analysis was rerun
to prepare it. Integrity checks read and compare the saved bytes.

The [Phishing, HTRU2 and Rice records](../tabular-classification/) remain in their
separate published release. This addition does not introduce other exploratory
tabular datasets.

## Contents and manuscript map

| Result | Archived records, relative to `workspace/` |
|---|---|
| Table II: 18 datasets and 180 universal-block fits | `output/prxq-universal-rerun-2026-10-15/` |
| Appendix G diagonal probe/loss controls: 30 new fits and 10 reused baselines | `output/prxq-diagonal-probe-loss-2026-10-16/` |
| Table III; repeated Hamiltonian and size comparisons; Figure 5(c) and Table V | `output/prxq-universal-rebuttal-2026-10-16/`: 290 fits, fixed splits, checkpoints, full trajectories and predictions |
| Appendix H, Figures 10–11 | The same campaign's `gradient-data/`, `gradients/` and case lists: 45 datasets, 9,000 initializations |
| Physical noise and finite measurements | The same campaign's `noise/`, original unitary/generator arrays, stream identities, per-model outputs and counts |
| Figure 4 Iris rerun | `manuscript-records/numerical_sources/iris_20260915/iris-run-records.tar.gz`: frozen source, inputs, 24 calibration fits and 40 final fits |
| Iris paired probe controls | `output/prxq-iris-probe-diagnostic-2026-10-16/`: 120 fits and its exact original Iris source/input copy |
| Unchanged moment baselines and source inputs | Selected records under `output/prxq-a6000-results-2026-09-08/evidence/` and `output/prxq-b10-experiments-2026-09-13/e1-raw/analysis/` |
| Earlier noise records retained for provenance and recovery | `legacy-noise-retention` entries in the manifest; these do not replace the current universal-block results |

`records-manifest.json` lists every file, byte count, SHA-256 and group.
`seed-index.json` locates the principal case lists and seed grids. The original
records retain the fitting/validation/test identities, model-selection rules,
data-generation keys, source hashes and measurement random streams.
`scope.json` records inclusion boundaries.

## Download, restore and verify

Use Python 3.12 or later. The restore utility uses only the standard library;
scientific dependencies are unnecessary for this step.

```sh
python restore.py download --out downloads
python restore.py verify --data downloads/workspace
```

The utility downloads the multipart archive, verifies each part and the assembled
archive, restores only safe relative paths, and checks every restored file.
Identical files share storage through internal hard links. The archive contains
`workspace/RECORDS.json`, whose checksum is authenticated by `release.json`.
Use a new destination when extracting. Downloads can resume from already verified
whole parts; incomplete parts are downloaded again.

The verification does not import archived Python modules, load pickled models,
fit a model, replay predictions or recalculate manuscript statistics. The original
verification reports are historical evidence and remain dated to their original
campaigns. See `verification.json` for the checks performed for this publication.

## Reproduction entrypoints

Work in a separate copy when running scientific code: some original entrypoints
write summaries or skip already completed runs. Preserve the extracted
`experiments/` and `output/` layout.

- Table II: `output/prxq-universal-rerun-2026-10-15/README.md`, `PROTOCOL.md`,
  `requirements.txt` and `replay.py` document the original portable package.
- The 290-fit campaign, gradients and noise: see
  `experiments/qrenn-universal-rebuttal-20261016/REPRODUCE.md`. It specifies
  separate model checks, analyses and training commands, with the original
  Python, NumPy, SciPy and PyTorch versions.
- Diagonal controls: `output/prxq-diagonal-probe-loss-2026-10-16/PROTOCOL.zh.md`,
  `freeze.json`, `run_diagnosis.py` and `environment.json` record the paired design.
- Iris controls: `output/prxq-iris-probe-diagnostic-2026-10-16/PROTOCOL.zh.md`,
  `freeze.json` and `run_probe_controls.py`; its `source/` holds the original
  implementation, dependencies, input folds and baseline model records.
- Retained older runs: `historical-source/versions.json` maps all 46 referenced
  source-file versions to exact bytes recovered from local Git history. The
  `by-sha256/` filenames distinguish different versions of the same module.
  Original source identity and the later maintenance checkout are separate.

Archived guides retain their historical table numbers, release-status wording
and plotting choices. The table above and `manuscript-records/numerical_sources/README.md`
map those records to the current manuscript. In particular, an earlier Figure 5
plotting script does not supply the current Figure 5(a) historical batch.
Frozen scientific files are unchanged. Absolute paths embedded in older records
are historical locators; use the corresponding relative archive paths.

## Historical records still to be supplied

Original per-run records for historical Figure 5(a) will be supplied separately
by the authors. They have not been received or verified for this release.
Figure 5(b) remains explicitly identified as digitized plot data. Available
historical scripts, plots or aggregate arrays must not be interpreted as recovery
of missing individual runs. The historical open-boundary SPT correspondence
question remains separate from the archived periodic-SPT rerun.

The older draft release `prxq-rebuttal-2026-09` is incomplete and is not required
to restore this package. This release is self-contained for the files listed in
its manifest; it does not claim to recover every historical experiment.
