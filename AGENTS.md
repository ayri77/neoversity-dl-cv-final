# Project instructions for coding agents

Kaggle competition "Deep Learning for CV and NLP 2026-10": a reduced PetFinder.my
task — predict `AdoptionSpeed` (classes 1–4) from the free-text `Description` and
the pet photos only (no tabular columns). Metric: quadratic weighted kappa (QWK).
The solution is complete (OOF QWK 0.621, public LB 0.8088); work now is polishing,
documentation and optional improvements. `CLAUDE.md` holds the detailed project
log (findings, results per notebook, pitfalls) — read it before larger changes.

## Layout
- `notebooks/01_eda … 07_fusion_final.ipynb` — research notebooks, one stage each,
  with analysis and ablations. `notebooks/final_submission.ipynb` — the single
  reproducible path from `data/raw` to `submissions/submission.csv`.
- `src/` — all reusable code; notebooks only orchestrate. `src/pipeline.py` wraps
  the stages with the final fixed parameters (`FinalConfig`).
- `scripts/run_nb.py` — execute a notebook in place with a live log in `logs/`.
- `docs/forum_post.md` — the solution write-up for the competition forum.
- Data (`data/`), caches, embeddings and submissions are not tracked by git.

## Conventions
- Notebook markdown is in Ukrainian; code, comments and docstrings in English.
- Environment: `uv sync`, `uv run …`, Jupyter kernel `neoversity-dl`. GPU has 12 GB:
  use bf16 autocast, size batches accordingly. Load transformer weights in fp32
  (`dtype=torch.float32`) — transformers 5 otherwise keeps fp16 checkpoints and AdamW diverges.
- Run long notebooks with `uv run python scripts/run_nb.py <notebook>`.
- `uv run pytest` must pass after changes to `src/`.

## Rules that protect the results
- Do not use the original 2019 PetFinder data or labels (leak). Ideas from public
  solutions are fine but must be credited in the write-up.
- Folds are fixed: `StratifiedKFold(5, shuffle=True, random_state=42)` on the train
  order from `load_data()` (column `fold`). Plain stratified folds are intentional —
  near-duplicate listings occur across the random train/test split in the same
  proportion, so grouped folds would under-estimate. Never change folds silently.
- Every target-dependent feature or level-1 model must be computed out-of-fold on
  these folds. Save level-1 outputs as OOF (train) + fold-averaged test predictions.
- Regression + `OptimizedRounder(labels=(1, 2, 3, 4))` for thresholds; do not shift labels.
- Read `PetID` as `str`; `load_data()` zero-pads 4 test ids for image lookup and
  `write_submission()` maps them back to the raw test.csv spelling — always use it.
- Kaggle public LB is a small, noisy split: select models by OOF QWK, report the
  honest estimate (thresholds fit on other folds) alongside.
- Do not commit, push, create remotes or post anything without an explicit user request.
