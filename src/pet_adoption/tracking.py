"""An optional local MLflow index over already completed file bundles."""

import json
from pathlib import Path

from pet_adoption.artifacts import read_run_bundle
from pet_adoption.config import flatten_config
from pet_adoption.paths import sqlite_uri


class TrackingError(RuntimeError):
    """Indexing failed; the authoritative bundle remains available."""


def log_run_bundle(bundle_path: Path, *, database: Path, artifact_root: Path) -> str:
    """Copy a completed bundle into a local SQLite-backed MLflow experiment.

    Each call creates an indexing run. Retry does not train or rewrite the source;
    a failed partial indexing run can remain in MLflow. No autologging is used.
    """
    run_id: str | None = None
    try:
        bundle = read_run_bundle(bundle_path)
        database = database.resolve()
        artifact_root = artifact_root.resolve()
        if database.is_relative_to(bundle.path) or artifact_root.is_relative_to(bundle.path):
            raise ValueError("MLflow storage must be separate from the source bundle.")
        database.parent.mkdir(parents=True, exist_ok=True)
        artifact_root.mkdir(parents=True, exist_ok=True)
        # Lazy import keeps all file persistence usable without MLflow.
        from mlflow import MlflowClient

        client = MlflowClient(tracking_uri=sqlite_uri(database), registry_uri=sqlite_uri(database))
        manifest = bundle.manifest
        experiment = client.get_experiment_by_name(manifest["experiment_name"])
        experiment_id = (
            client.create_experiment(
                manifest["experiment_name"], artifact_location=artifact_root.as_uri()
            )
            if experiment is None
            else experiment.experiment_id
        )
        if experiment is not None:
            if experiment.lifecycle_stage != "active":
                raise ValueError("The MLflow experiment is deleted; restore it explicitly.")
            if experiment.artifact_location != artifact_root.as_uri():
                raise ValueError("Existing experiment uses a different artifact location.")
        tags = {
            key: str(manifest.get(key) if manifest.get(key) is not None else "unknown")
            for key in (
                "run_id",
                "model_family",
                "feature_set",
                "dataset_id",
                "folds_id",
                "git_commit",
                "run_type",
            )
        }
        tags["modalities"] = json.dumps(manifest["modalities"])
        tags["mlflow.runName"] = manifest["run_id"]
        run_id = client.create_run(experiment_id, tags=tags).info.run_id
        for key, value in flatten_config(bundle.config).items():
            client.log_param(run_id, key, json.dumps(value, ensure_ascii=False, allow_nan=False))
        for key, value in bundle.metrics.items():
            client.log_metric(run_id, key, value)
        client.log_artifact(run_id, str(bundle.path / "manifest.json"), artifact_path="bundle")
        for relative in manifest["artifacts"].values():
            client.log_artifact(run_id, str(bundle.path / relative), artifact_path="bundle")
        client.set_terminated(run_id, status="FINISHED")
        return run_id
    except Exception as error:
        if run_id is not None:
            try:
                client.set_terminated(run_id, status="FAILED")
            except Exception:
                pass  # Preserve the original tracking error if storage is unavailable.
        raise TrackingError(
            f"MLflow indexing failed for bundle {bundle_path}. "
            f"The file bundle is unchanged and can be read/reindexed. "
            f"MLflow run ID: {run_id or 'not created'}. Cause: {error}"
        ) from error
