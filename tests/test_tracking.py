import hashlib

import pytest
from mlflow import MlflowClient

from pet_adoption.artifacts import RunMetadata, read_run_bundle, save_run_bundle
from pet_adoption.config import flatten_config, load_config
from pet_adoption.paths import artifact_path, sqlite_uri
from pet_adoption.tracking import TrackingError, log_run_bundle


def make_bundle(root, run_id):
    return save_run_bundle(
        root / "runs",
        metadata=RunMetadata(
            "tracking-test",
            "synthetic",
            "none",
            (),
            42,
            dataset_id="test-data",
            folds_id="test-folds",
            run_type="smoke",
        ),
        config={"seed": 42, "thresholds": {"method": "test"}, "classes": [0, 1]},
        metrics={"qwk_calibration": 0.75},
        run_id=run_id,
    )


def digest_files(path):
    return {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in path.iterdir()}


def test_sqlite_tracking_and_reuse_experiment(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_TELEMETRY", "false")
    database, artifacts = tmp_path / "mlflow" / "test.db", tmp_path / "mlflow" / "files"
    bundle = make_bundle(tmp_path, "first")
    original = digest_files(bundle)
    run_id = log_run_bundle(bundle, database=database, artifact_root=artifacts)
    client = MlflowClient(tracking_uri=sqlite_uri(database))
    run = client.get_run(run_id)
    assert database.is_file()
    assert run.info.status == "FINISHED"
    assert run.data.metrics["qwk_calibration"] == 0.75
    assert run.data.params["thresholds.method"] == '"test"'
    assert run.data.tags["dataset_id"] == "test-data"
    assert run.data.tags["folds_id"] == "test-folds"
    assert run.data.tags["model_family"] == "none"
    assert run.data.tags["feature_set"] == "synthetic"
    assert run.data.tags["modalities"] == "[]"
    assert run.data.tags["git_commit"] == "unknown"
    assert run.data.tags["run_id"] == "first"
    assert {item.path for item in client.list_artifacts(run_id, "bundle")} == {
        "bundle/manifest.json",
        "bundle/config.yaml",
        "bundle/metrics.json",
    }
    assert digest_files(bundle) == original
    second = log_run_bundle(
        make_bundle(tmp_path, "second"), database=database, artifact_root=artifacts
    )
    assert client.get_run(second).info.experiment_id == run.info.experiment_id


def test_tracking_failure_preserves_bundle(tmp_path, monkeypatch):
    bundle = make_bundle(tmp_path, "failure")
    original = digest_files(bundle)

    def fail(*args, **kwargs):
        raise OSError("simulated MLflow outage")

    monkeypatch.setattr(MlflowClient, "log_metric", fail)
    with pytest.raises(TrackingError, match="bundle is unchanged"):
        log_run_bundle(bundle, database=tmp_path / "db.sqlite", artifact_root=tmp_path / "files")
    assert digest_files(bundle) == original
    assert read_run_bundle(bundle).metrics["qwk_calibration"] == 0.75


def test_invalid_storage_does_not_touch_source(tmp_path):
    bundle = make_bundle(tmp_path, "bad-storage")
    before = digest_files(bundle)
    with pytest.raises(TrackingError, match="separate"):
        log_run_bundle(bundle, database=bundle / "db.sqlite", artifact_root=tmp_path / "files")
    assert digest_files(bundle) == before


def test_config_and_portable_paths(tmp_path):
    path = tmp_path / "base.yaml"
    path.write_text("seed: 42\npaths:\n  runs: artifacts/runs\n", encoding="utf-8")
    config = load_config(path)
    assert flatten_config(config) == {"seed": 42, "paths.runs": "artifacts/runs"}
    assert artifact_path(tmp_path, "artifacts/runs") == tmp_path / "artifacts" / "runs"
    for invalid in ["../outside", "elsewhere/runs", "C:/outside", "/tmp/outside"]:
        with pytest.raises(ValueError):
            artifact_path(tmp_path, invalid)
    with pytest.raises(ValueError):
        flatten_config({"a.b": 1})
