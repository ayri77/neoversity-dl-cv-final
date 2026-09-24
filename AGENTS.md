# Project instructions

- Create or modify files only inside this project directory.
- Write code, comments, configuration names, and documentation in English.
- `ayri77/neoversity-ml-final` is always a read-only reference, including branch
  `feature/prepared-dataset-pipeline-v1`. Never modify its files or GitHub state.
- The official model-selection metric is Quadratic Weighted Kappa (QWK).
- Never silently change folds, target mapping, the ordered class space, metric
  definitions, seeds, or threshold methodology. Record and explain changes.
- Every real experiment must save config, metrics, fold identity, OOF predictions,
  and test predictions. Preserve raw continuous scores as well as derived labels.
- Validate OOF and test row identity and order against the authoritative input
  order before saving. Never repair alignment by silently sorting predictions.
- Kaggle Public Score is an external signal, never the sole selection criterion.
- Maintain one authoritative experiment execution path. Stage 1 has only a
  synthetic tracking smoke entry point; introduce the real path in a later stage.
- File run bundles are authoritative. Complete them before attempting MLflow
  indexing. Tracking failures must not invalidate or remove completed bundles.
- Do not overwrite historical runs by default.
- No custom Web UI, Streamlit control panel, deployment infrastructure, AutoML,
  parallel runners, or compatibility framework without a separate user decision.
- Do not implement competition loaders, text models, or image models in Stage 1.
- After the competition starts, change infrastructure only to fix blocking issues.
- The final `.ipynb` must be reproducible and as self-contained as practical.
- Use uv and the project `.venv`. Keep caches and temporary outputs in the project.
- Tests must use isolated temporary storage, never historical run state.
- Run pytest, Ruff, mypy, and the tracking smoke test when changing the foundation.
- Do not commit or push without an explicit user request. Do not create a remote.
