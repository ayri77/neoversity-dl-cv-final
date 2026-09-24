import json
import shutil
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from pet_adoption.artifacts import (
    RunMetadata,
    read_run_bundle,
    save_run_bundle,
    validate_predictions,
)


@pytest.fixture
def metadata():
    return RunMetadata("test", "synthetic", "none", (), 42, run_type="smoke")


@pytest.fixture
def predictions():
    return pd.DataFrame(
        {
            "row_id": ["a", "b", "c"],
            "score": [0.1, 1.1, 2.1],
            "prediction": [0, 1, 2],
            "target": [0, 1, 2],
            "fold": [0, 1, 0],
        }
    )


def test_complete_bundle_round_trip_and_relocation(tmp_path, metadata, predictions):
    path = save_run_bundle(
        tmp_path / "runs",
        metadata=replace(metadata, run_type="experiment", folds_id="fixed-v1"),
        config={"classes": [0, 1, 2]},
        metrics={"qwk": 1.0},
        run_id="roundtrip",
        oof_predictions=predictions,
        test_predictions=predictions.drop(columns=["target", "fold"]),
        expected_oof_row_ids=["a", "b", "c"],
        expected_test_row_ids=["a", "b", "c"],
        thresholds={"cutpoints": [0.5, 1.5]},
        fold_metrics=pd.DataFrame({"fold": [0], "qwk": [1]}),
    )
    relocated = tmp_path / "relocated"
    shutil.copytree(path, relocated)
    bundle = read_run_bundle(relocated)
    assert bundle.manifest["run_id"] == "roundtrip"
    assert bundle.manifest["folds_id"] == "fixed-v1"
    assert bundle.manifest["git_commit"] is None
    assert bundle.metrics == {"qwk": 1.0}
    pd.testing.assert_frame_equal(bundle.oof_predictions, predictions)
    assert bundle.test_predictions["row_id"].tolist() == ["a", "b", "c"]
    assert bundle.thresholds == {"cutpoints": [0.5, 1.5]}
    assert bundle.fold_metrics["qwk"].tolist() == [1]
    assert all(not value.startswith(("/", "C:")) for value in bundle.manifest["artifacts"].values())


def test_minimal_smoke_bundle(tmp_path, metadata):
    path = save_run_bundle(tmp_path, metadata=metadata, config={}, metrics={})
    assert read_run_bundle(path).oof_predictions is None
    assert set(item.name for item in path.iterdir()) == {
        "manifest.json",
        "config.yaml",
        "metrics.json",
    }


def test_experiment_requires_predictions_and_fold_identity(tmp_path, metadata):
    with pytest.raises(ValueError, match="Experiments require"):
        save_run_bundle(
            tmp_path, metadata=replace(metadata, run_type="experiment"), config={}, metrics={}
        )


def test_no_overwrite_without_flag(tmp_path, metadata):
    path = save_run_bundle(
        tmp_path, metadata=metadata, config={}, metrics={"qwk": 0}, run_id="same"
    )
    with pytest.raises(FileExistsError):
        save_run_bundle(tmp_path, metadata=metadata, config={}, metrics={"qwk": 1}, run_id="same")
    assert read_run_bundle(path).metrics["qwk"] == 0
    save_run_bundle(
        tmp_path, metadata=metadata, config={}, metrics={"qwk": 1}, run_id="same", overwrite=True
    )
    assert read_run_bundle(path).metrics["qwk"] == 1


def test_write_failure_preserves_existing_bundle(tmp_path, metadata, monkeypatch):
    path = save_run_bundle(
        tmp_path, metadata=metadata, config={}, metrics={"qwk": 0}, run_id="same"
    )
    before = {item.name: item.read_bytes() for item in path.iterdir()}

    def fail(*args, **kwargs):
        raise OSError("simulated disk error")

    monkeypatch.setattr("pet_adoption.artifacts._json", fail)
    with pytest.raises(OSError, match="disk error"):
        save_run_bundle(
            tmp_path,
            metadata=metadata,
            config={},
            metrics={"qwk": 1},
            run_id="same",
            overwrite=True,
        )
    assert {item.name: item.read_bytes() for item in path.iterdir()} == before
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("ids", [["b", "a", "c"], ["a", "b"], ["a", "a", "c"]])
def test_row_alignment_rejected(predictions, ids):
    with pytest.raises(ValueError):
        validate_predictions(predictions, ids, classes=[0, 1, 2], oof=True)


@pytest.mark.parametrize(
    "column,value", [("score", np.nan), ("prediction", 9), ("fold", -1), ("row_id", None)]
)
def test_corrupt_predictions_rejected(predictions, column, value):
    predictions.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_predictions(predictions, ["a", "b", "c"], classes=[0, 1, 2], oof=True)


def test_writer_requires_reference_ids(tmp_path, metadata, predictions):
    with pytest.raises(ValueError, match="Expected row IDs"):
        save_run_bundle(
            tmp_path,
            metadata=metadata,
            config={"classes": [0, 1, 2]},
            metrics={},
            oof_predictions=predictions,
        )


@pytest.mark.parametrize("run_id", ["../escape", "CON", "a/b", "", "a.b"])
def test_unsafe_run_ids(tmp_path, metadata, run_id):
    with pytest.raises(ValueError):
        save_run_bundle(tmp_path, metadata=metadata, config={}, metrics={}, run_id=run_id)


def test_nonfinite_metric_rejected(tmp_path, metadata):
    with pytest.raises(ValueError, match="finite"):
        save_run_bundle(tmp_path, metadata=metadata, config={}, metrics={"qwk": float("nan")})


def test_missing_artifact_rejected(tmp_path, metadata):
    path = save_run_bundle(tmp_path, metadata=metadata, config={}, metrics={})
    (path / "metrics.json").unlink()
    with pytest.raises(ValueError, match="missing"):
        read_run_bundle(path)


def test_reader_rejects_inventory_escape(tmp_path, metadata):
    path = save_run_bundle(tmp_path, metadata=metadata, config={}, metrics={})
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["metrics"] = "../escape.json"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="invalid"):
        read_run_bundle(path)
