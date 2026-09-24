"""Portable run bundles published only after all files are written successfully."""

import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml

from pet_adoption.metrics import encode_labels
from pet_adoption.thresholds import validate_cutpoints


@dataclass(frozen=True)
class RunMetadata:
    experiment_name: str
    feature_set: str
    model_family: str
    modalities: tuple[str, ...]
    seed: int
    dataset_id: str | None = None
    dataset_hash: str | None = None
    folds_id: str | None = None
    folds_hash: str | None = None
    git_commit: str | None = None
    run_type: Literal["experiment", "smoke"] = "experiment"


@dataclass(frozen=True)
class RunBundle:
    path: Path
    manifest: dict[str, Any]
    config: dict[str, Any]
    metrics: dict[str, float]
    oof_predictions: pd.DataFrame | None
    test_predictions: pd.DataFrame | None
    fold_metrics: pd.DataFrame | None
    thresholds: dict[str, Any] | None


def get_git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def validate_predictions(
    frame: pd.DataFrame,
    expected_row_ids: Sequence[Any],
    *,
    classes: Sequence[Any],
    oof: bool,
) -> None:
    """Require exact input order, unique IDs, complete scores, and ordinal labels."""
    required = {"row_id", "score", "prediction"} | ({"target", "fold"} if oof else set())
    if not frame.columns.is_unique or not required.issubset(frame.columns):
        raise ValueError(f"Prediction columns must be unique and include {sorted(required)}.")
    expected = pd.Index(expected_row_ids)
    actual = pd.Index(frame["row_id"])
    if len(expected) == 0 or expected.has_duplicates or expected.isna().any():
        raise ValueError("Expected row IDs must be nonempty, unique, and non-null.")
    if actual.has_duplicates or actual.isna().any() or not actual.equals(expected):
        raise ValueError("Prediction row alignment differs from expected row IDs/order.")
    if not pd.api.types.is_numeric_dtype(frame["score"]):
        raise ValueError("Prediction scores must be numeric.")
    if not np.isfinite(frame["score"].to_numpy(dtype=float)).all():
        raise ValueError("Prediction scores must be finite.")
    encode_labels(frame["prediction"].to_numpy(), classes)
    if oof:
        encode_labels(frame["target"].to_numpy(), classes)
        folds = frame["fold"]
        if not pd.api.types.is_integer_dtype(folds) or folds.isna().any() or (folds < 0).any():
            raise ValueError("OOF fold assignments must be non-null, nonnegative integers.")


def _metrics(values: Mapping[str, float]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key or isinstance(value, (bool, str)):
            raise ValueError("Metrics must have nonempty names and finite numeric values.")
        result[key] = float(value)
        if not np.isfinite(result[key]):
            raise ValueError("Metrics must be finite; undefined QWK cannot be persisted.")
    return result


def _json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def save_run_bundle(
    runs_root: Path,
    *,
    metadata: RunMetadata,
    config: Mapping[str, Any],
    metrics: Mapping[str, float],
    run_id: str | None = None,
    oof_predictions: pd.DataFrame | None = None,
    test_predictions: pd.DataFrame | None = None,
    expected_oof_row_ids: Sequence[Any] | None = None,
    expected_test_row_ids: Sequence[Any] | None = None,
    fold_metrics: pd.DataFrame | None = None,
    thresholds: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Stage a complete bundle, then rename it into place on the same filesystem.

    A per-run exclusive lock prevents competing writers. Explicit replacement
    retains the previous directory until publication succeeds. MLflow is never
    involved. Prediction-less bundles are allowed only for synthetic smoke runs.
    """
    now = datetime.now(UTC)
    run_id = f"{now:%Y%m%dT%H%M%S%fZ}_{uuid4().hex[:8]}" if run_id is None else run_id
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,119}", run_id) or re.fullmatch(
        r"(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])", run_id
    ):
        raise ValueError("run_id must be a portable filename using letters, digits, - or _.")
    if metadata.run_type not in {"experiment", "smoke"}:
        raise ValueError("run_type must be experiment or smoke.")
    if metadata.run_type == "experiment" and (
        oof_predictions is None
        or test_predictions is None
        or not (metadata.folds_id or metadata.folds_hash)
    ):
        raise ValueError(
            "Experiments require OOF predictions, test predictions, and fold identity."
        )
    normalized_metrics = _metrics(metrics)
    for frame, expected, oof in [
        (oof_predictions, expected_oof_row_ids, True),
        (test_predictions, expected_test_row_ids, False),
    ]:
        if frame is not None:
            if expected is None:
                raise ValueError("Expected row IDs are required when saving predictions.")
            validate_predictions(frame, expected, classes=config.get("classes", []), oof=oof)
    if thresholds is not None:
        validate_cutpoints(thresholds["cutpoints"], n_classes=len(config.get("classes", [])))

    root = runs_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / run_id
    lock = root / f".{run_id}.lock"
    # Do not unlink a lock owned by another writer if exclusive creation fails.
    with lock.open("x", encoding="utf-8"):
        pass
    staging: Path | None = None
    backup: Path | None = None
    published = False
    try:
        if destination.is_symlink() or destination.is_junction():
            raise ValueError("A run destination cannot be a link or junction.")
        if destination.exists() and not overwrite:
            raise FileExistsError(
                f"Run already exists: {destination}; use overwrite=True explicitly."
            )
        staging = Path(tempfile.mkdtemp(prefix=f".{run_id}-", dir=root))
        (staging / "config.yaml").write_text(
            yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        _json(staging / "metrics.json", normalized_metrics)
        for name, frame in [
            ("oof_predictions", oof_predictions),
            ("test_predictions", test_predictions),
        ]:
            if frame is not None:
                frame.to_parquet(staging / f"{name}.parquet", index=False)
        if fold_metrics is not None:
            fold_metrics.to_csv(staging / "fold_metrics.csv", index=False)
        if thresholds is not None:
            _json(staging / "thresholds.json", dict(thresholds))
        inventory = {path.stem: path.name for path in sorted(staging.iterdir())}
        manifest = {
            "run_id": run_id,
            "created_at": now.isoformat(),
            "status": "completed",
            **asdict(metadata),
            "artifacts": inventory,
        }
        _json(staging / "manifest.json", manifest)
        for path in staging.iterdir():
            with path.open("r+b") as stream:
                os.fsync(stream.fileno())
        if destination.exists():
            backup = root / f".{run_id}-backup-{uuid4().hex}"
            destination.rename(backup)
        try:
            staging.rename(destination)
            published = True
        except BaseException:
            if backup is not None:
                backup.rename(destination)
                backup = None
            raise
        return destination
    finally:
        for temporary in (staging, backup if published else None):
            if temporary is not None and temporary.exists():
                # Only directories allocated under this runs root are removed.
                if temporary.parent != root or not temporary.name.startswith(f".{run_id}-"):
                    raise RuntimeError("Refusing cleanup outside the owned run staging paths.")
                shutil.rmtree(temporary)
        lock.unlink()


def read_run_bundle(path: Path) -> RunBundle:
    """Read published portable files independently of MLflow or the original root."""
    path = path.resolve()
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("Only completed bundles can be read or indexed.")
    inventory = manifest["artifacts"]
    for key, relative in inventory.items():
        candidate = path / relative
        if (
            not isinstance(key, str)
            or Path(relative).is_absolute()
            or not candidate.resolve().is_relative_to(path)
            or not candidate.is_file()
        ):
            raise ValueError("Bundle inventory contains an invalid or missing artifact path.")
    with (path / inventory["config"]).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    metrics = _metrics(json.loads((path / inventory["metrics"]).read_text(encoding="utf-8")))

    def predictions(name: str) -> pd.DataFrame | None:
        return pd.read_parquet(path / inventory[name]) if name in inventory else None

    return RunBundle(
        path,
        manifest,
        config,
        metrics,
        predictions("oof_predictions"),
        predictions("test_predictions"),
        pd.read_csv(path / inventory["fold_metrics"]) if "fold_metrics" in inventory else None,
        json.loads((path / inventory["thresholds"]).read_text(encoding="utf-8"))
        if "thresholds" in inventory
        else None,
    )
