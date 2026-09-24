# Minimal experiment contract

A real experiment saves one authoritative directory:

```text
artifacts/runs/<run_id>/
  manifest.json
  config.yaml
  metrics.json
  fold_metrics.csv           # optional
  oof_predictions.parquet    # required for real experiments
  test_predictions.parquet   # required for real experiments
  thresholds.json            # required when thresholds were applied/fitted
  plots/                     # optional, reserved for future reports
```

`RunMetadata.run_type="experiment"` requires both prediction tables and a folds
identifier/hash. Synthetic smoke bundles may omit predictions. Stage 1's smoke
does save both tables, explicitly marked as synthetic, without claiming real OOF
evaluation. The generic file writer does not train models or execute folds.

The manifest records `run_id`, UTC `created_at`, `status`, `experiment_name`,
`git_commit` (null without a commit), dataset ID/hash, folds ID/hash, feature set,
model family, ordered modalities, seed, run type, and relative artifact paths.
`completed` means the file bundle has been published, independently of whether
MLflow indexing succeeded. Incomplete staging directories are not completed runs.
The manifest is the entry point; its inventory lists every other saved file.

`config.yaml` contains the resolved configuration, including ordered classes,
target meaning, feature choices, model parameters, seeds, CV protocol, prediction
aggregation, and threshold/evaluation methodology when implemented. Keep secrets
out of configuration: the complete config is saved and indexed. Archive the
source revision and `uv.lock` with any future reproducibility handoff. An
uncommitted foundation smoke run has `git_commit=null` and is not a frozen code
revision. Stage 1 does not snapshot source or environments inside each bundle.

`metrics.json` is a flat map of names to finite numeric values; higher QWK is
better. Use distinct names for fixed-threshold, calibration, and honest evaluation
scores. Do not persist undefined QWK as zero. Fold metrics, if available, include
fold IDs and explicitly named scores. Public Kaggle scores are external evidence,
not substitutes for a fixed honest evaluation protocol.

## Prediction rows

Both tables require `row_id`, `score`, and `prediction`. OOF additionally requires
`target` and `fold` (nonnegative integer). `score` is the raw finite continuous
ordinal score; `prediction` and `target` belong to the full ordered class space.
Rows have unique non-null IDs and exactly match the authoritative reference ID
sequence supplied to the writer. Missing, extra, duplicate, or reordered rows
are rejected. Pandas index is not saved as implicit row identity.

Callers must supply the original train/test ID sequences independently from
prediction construction. Do not use prediction IDs as their own reference in
real experiments. Preserve the target and fold association throughout inference;
the Stage 1 writer validates structure/alignment but cannot prove leakage-free
training or authenticate assignments without the future CV/data adapter.

One row per training example is the initial contract. Repeated-CV aggregation
would need an explicit future decision. No submission schema is assumed yet.

## Thresholds and evaluation

For K classes, store K-1 finite strictly increasing cutpoints and the complete
ordered class list in config. Equality at a cutpoint maps to the higher class.
Application, fitting, and QWK evaluation are separate functions. Numeric label
values can be nonconsecutive; weights use their positions in the declared order.

When fitting thresholds, record method, initial thresholds or initialization rule,
candidate budget, pass limit, resulting cutpoints, convergence, fitting data scope,
and calibration QWK. Stage 1 defaults to evenly spaced initialization across the
score range and `ordered-coordinate-search-v1`. Saving thresholds is explicit:
pass the fitting result/methodology as the `thresholds` argument to the writer.

Optimizing thresholds and reporting QWK on the same labels is in-sample
calibration. Even when model scores are OOF, those labels influenced the
thresholds. Honest cross-fitted threshold evaluation is Stage 3 and is not
implemented. Never silently change folds, target mapping, class order, QWK,
threshold policy, or the meaning of a metric name.

## Persistence and indexing

Validate inputs, write a staging directory, flush files, and publish the completed
bundle before calling MLflow. Existing run IDs raise `FileExistsError` unless
`overwrite=True` is explicitly supplied. Prefer a new ID for every experiment.
After competition start, infrastructure changes are limited to blocking fixes.

MLflow receives flattened config, finite metrics, provenance tags, and copies of
the bundle files. It is an index/UI, not a runner or source of experiment truth.
Failure leaves the bundle readable through `read_run_bundle()` or standard
JSON/YAML/Parquet tools. A retry may create another MLflow run; no database
registry or artifact migration system is included.
