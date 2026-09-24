# Foundation Stage 1

The research path is deliberately small:

```text
resolved config + synthetic scores
  -> fit_thresholds -> apply_thresholds -> quadratic_weighted_kappa
  -> save_run_bundle -> read_run_bundle -> log_run_bundle
                         |                  |
                         portable files     local SQLite + artifact copies
```

`scripts/run_tracking_smoke_test.py` is the only execution entry point today.
It is a plumbing check, not a model runner. Later, one experiment execution path
will call the same persistence functions. Future notebooks analyze bundles and
document evidence; they must not create competing implementations of metrics or CV.

- `config.py`: YAML loading and config flattening, no competition schema.
- `paths.py`: explicit root plus relative paths under `artifacts/`.
- `metrics.py`: validated rank encoding and scikit-learn QWK with full class space.
- `thresholds.py`: application and constrained coordinate calibration.
- `artifacts.py`: small dataclasses, alignment validation, stage/publish/read.
- `tracking.py`: local MLflow client and error boundary after bundle publication.

There is no dataset registry, training framework, compatibility layer, source
adapter registry, deployment, custom UI, or model code. The final data adapter
must wait for the real dataset and competition rules.

## Initial read-only audit (2026-09-03)

The new directory was empty and was not a Git repository. No applicable local
`AGENTS.md` was found in the project or its ancestors. Git was initialized on
`main`, without a remote or commit. No old checkout existed at the inspected
adjacent `neoversity-ml-final` locations; the old project was inspected through
GitHub only, without cloning it.

Reference: [ayri77/neoversity-ml-final at 7acc6f1](https://github.com/ayri77/neoversity-ml-final/tree/7acc6f121d631f1a1b88b8c660baf677c7994078),
branch `feature/prepared-dataset-pipeline-v1`.

Read only:

- `AGENTS.md`.
- `docs/current-project-status.md`.
- `docs/final-audit.md`.
- `src/churn_ml/run_artifacts.py` (experiment outputs and prediction contract).
- `src/churn_ml/experiment_v2_contract.py` and `experiment_v2_schema.py`
  (explicit identity and ordered schema concepts).
- `docs/mlflow-local-index.md` (filesystem authority and MLflow separation).

Safely reused ideas: persist resolved configuration and provenance with results;
retain row identity/order and raw predictions; never silently overwrite runs;
index completed filesystem evidence in MLflow without making it authoritative.
No old implementation was copied.

The old audit reported a full test suite affected by persisted-state mismatches
and UI timeouts. This foundation instead tests fresh isolated storage. The old
Streamlit UI, AutoGluon paths, dataset/candidate registries, campaign runners,
repeated/nested research frameworks, deployment, and compatibility machinery
were deliberately not transferred.

## Implementation choices and limits

Class order is required explicitly; rank distance, not numeric label spacing,
defines quadratic penalties. QWK for identical constant labels is undefined and
raises `ValueError`, rather than writing NaN or declaring perfect agreement.

Threshold calibration evaluates a bounded grid of observed scores plus the
initial cutpoints and a value above the maximum. Every candidate stays strictly
between its neighbors. Ties keep the current thresholds; no final sorting is
performed. The search is deterministic and may reach a local optimum. The
candidate budget and pass limit are recorded. Constant scores require explicit
initialization; constant targets cannot calibrate thresholds.

Bundles use same-filesystem staging, fsync, a per-run exclusive writer lock,
and directory rename. Normal publication has no visible partial bundle.
Explicit replacement temporarily renames the old directory to a backup and
restores it if publication fails. A process/power failure can leave a lock,
staging directory, or backup: inspect it manually before removing anything.
Power-loss transactions and concurrent readers during explicit replacement are
not guaranteed. Defaults preserve historical results.

Stage 1 records supplied dataset/fold identifiers and hashes; it does not define
competition hashing or prove out-of-fold provenance. OOF fold assignments are
stored in the prediction table. The future adapter/CV stage must generate and
verify these identities from real data.

Primary API references:
[scikit-learn QWK](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html),
[MLflow local SQLite tracking](https://www.mlflow.org/docs/latest/ml/tracking/tutorials/local-database/),
[MLflow client](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html).

## Stage 1 validation (2026-09-03)

- `uv sync`: passed; 127 packages resolved, 123 installed for Windows/Python 3.12.12.
- `uv run pytest`: 60 passed in 17.98 seconds.
- `uv run ruff check .`: passed.
- `uv run mypy src`: passed, seven source modules checked in strict mode.
- `uv run python scripts/run_tracking_smoke_test.py`: passed, no training.
- `uv run mlflow ui --help`: confirmed the documented CLI flags; no UI server started.
- `git diff --check`: passed; untracked text files also checked for trailing whitespace.

Smoke bundle: `20260903T063556587613Z_bf8c58ab`.
MLflow run: `ca522719d5074130b07d943d51978fbd`.
Synthetic calibration QWK: `0.8974358974358975`.
Cutpoints: `[0.5237928863687162, 1.7182664990106544]`.

The bundle was read without importing MLflow. All 60 synthetic train rows and
12 test rows retained their expected order. A read-only SQLite query confirmed
`FINISHED`, matching metrics/tags, and byte-identical copies of all six bundle
files. All generated run/MLflow files are Git-ignored. Source/config/docs contain
no absolute path to the old project. The reference branch still points to
`7acc6f121d631f1a1b88b8c660baf677c7994078`; only read operations were used.

The initial network-isolated `uv sync` could not access PyPI; the authorized retry
succeeded. Restricted Windows sandbox ACLs blocked pytest temporary directories;
validation succeeded in the normal Windows context with all temporary files
inside this project. The first functional run found Windows fsync requiring a
writable file descriptor; this was fixed and the full suite passed afterward.
Float assertions use tolerance, and optimizer tests allow a local optimum while
checking improvement, ordering, determinism, and independently recomputed QWK.

Kaggle execution and honest model-selection performance remain untested. No
commit, remote, push, legacy modification, model training, or Stage 2-9 work was
performed.
