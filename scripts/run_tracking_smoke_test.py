"""Exercise ordinal calibration and tracking without training any model."""

import argparse
import hashlib
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from pet_adoption.artifacts import RunMetadata, get_git_commit, read_run_bundle, save_run_bundle
from pet_adoption.config import load_config
from pet_adoption.metrics import quadratic_weighted_kappa
from pet_adoption.paths import artifact_path
from pet_adoption.thresholds import apply_thresholds, fit_thresholds
from pet_adoption.tracking import log_run_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    config = load_config(config_path)
    if config["evaluation"] != {"metric": "qwk", "scope": "synthetic-in-sample-calibration"}:
        raise ValueError("The smoke test only supports synthetic in-sample QWK calibration.")
    if (
        config["thresholds"]["method"] != "ordered-coordinate-search-v1"
        or config["thresholds"]["initialization"] != "evenly-spaced-score-range"
    ):
        raise ValueError("Unsupported smoke threshold methodology; update it explicitly.")
    os.environ.setdefault("MLFLOW_ENABLE_TELEMETRY", "false")
    rng = np.random.default_rng(config["seed"])
    classes = config["classes"]
    ranks = np.tile(np.arange(len(classes)), 20)
    labels = np.asarray(classes)[ranks]
    scores = ranks.astype(float) + rng.normal(0, 0.55, size=len(ranks))
    fit = fit_thresholds(
        labels,
        scores,
        classes=classes,
        max_candidates=config["thresholds"]["max_candidates"],
        max_passes=config["thresholds"]["max_passes"],
    )
    prediction = apply_thresholds(scores, fit.cutpoints, classes=classes)
    qwk = quadratic_weighted_kappa(labels, prediction, classes=classes)
    row_ids = [f"synthetic-train-{index}" for index in range(len(labels))]
    test_ids = [f"synthetic-test-{index}" for index in range(12)]
    test_scores = rng.uniform(-0.5, len(classes) - 0.5, size=len(test_ids))
    # These assignments test serialization only; no CV or OOF model was run.
    fold = np.arange(len(labels)) % 3
    oof = pd.DataFrame(
        {
            "row_id": row_ids,
            "target": labels,
            "score": scores,
            "prediction": prediction,
            "fold": fold,
        }
    )
    test = pd.DataFrame(
        {
            "row_id": test_ids,
            "score": test_scores,
            "prediction": apply_thresholds(test_scores, fit.cutpoints, classes=classes),
        }
    )
    metadata = RunMetadata(
        experiment_name=config["experiment_name"],
        feature_set=config["feature_set"],
        model_family=config["model_family"],
        modalities=tuple(config["modalities"]),
        seed=config["seed"],
        dataset_id=config["dataset_id"],
        folds_id=config["folds_id"],
        dataset_hash=hashlib.sha256(oof.to_json(orient="split", index=False).encode()).hexdigest(),
        folds_hash=hashlib.sha256(fold.astype("<i8").tobytes()).hexdigest(),
        git_commit=get_git_commit(project_root),
        run_type="smoke",
    )
    paths = config["paths"]
    bundle_path = save_run_bundle(
        artifact_path(project_root, paths["runs"]),
        metadata=metadata,
        config=config,
        metrics={"qwk_calibration": qwk},
        oof_predictions=oof,
        test_predictions=test,
        expected_oof_row_ids=row_ids,
        expected_test_row_ids=test_ids,
        thresholds={**asdict(fit), "evaluation_scope": config["evaluation"]["scope"]},
    )
    bundle = read_run_bundle(bundle_path)
    print(f"Run ID: {bundle.manifest['run_id']}", flush=True)
    print(f"QWK (synthetic in-sample calibration): {qwk:.6f}", flush=True)
    print(f"Thresholds: {list(fit.cutpoints)}", flush=True)
    print(f"Artifacts: {bundle_path}", flush=True)
    mlflow_run_id = log_run_bundle(
        bundle_path,
        database=artifact_path(project_root, paths["mlflow_database"]),
        artifact_root=artifact_path(project_root, paths["mlflow_artifacts"]),
    )
    print(f"MLflow run ID: {mlflow_run_id}")


if __name__ == "__main__":
    main()
