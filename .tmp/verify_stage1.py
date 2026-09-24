"""One-off read-only validation of the Stage 1 deliverables."""

import hashlib
import json
import re
import sqlite3
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

from pet_adoption.artifacts import read_run_bundle
from pet_adoption.metrics import quadratic_weighted_kappa

root = Path.cwd()
run_id = "20260903T063556587613Z_bf8c58ab"
mlflow_id = "ca522719d5074130b07d943d51978fbd"
source = root / "artifacts" / "runs" / run_id
assert "mlflow" not in sys.modules
bundle = read_run_bundle(source)
assert "mlflow" not in sys.modules
assert bundle.manifest["status"] == "completed"
assert bundle.oof_predictions["row_id"].tolist() == [f"synthetic-train-{i}" for i in range(60)]
assert bundle.test_predictions["row_id"].tolist() == [f"synthetic-test-{i}" for i in range(12)]
assert bundle.metrics["qwk_calibration"] == quadratic_weighted_kappa(
    bundle.oof_predictions["target"], bundle.oof_predictions["prediction"],
    classes=bundle.config["classes"],
)
database = root / "artifacts" / "mlflow" / "mlflow.db"
with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
    status, artifact_uri = connection.execute(
        "SELECT status, artifact_uri FROM runs WHERE run_uuid = ?", (mlflow_id,)
    ).fetchone()
    tags = dict(connection.execute("SELECT key, value FROM tags WHERE run_uuid = ?", (mlflow_id,)))
    metrics = dict(connection.execute(
        "SELECT key, value FROM metrics WHERE run_uuid = ?", (mlflow_id,)
    ))
assert status == "FINISHED"
assert tags["run_id"] == run_id
assert metrics == bundle.metrics
copied_bundle = Path(url2pathname(urlparse(artifact_uri).path)) / "bundle"
files = ["manifest.json", *bundle.manifest["artifacts"].values()]
for relative in files:
    assert hashlib.sha256((source / relative).read_bytes()).digest() == hashlib.sha256(
        (copied_bundle / relative).read_bytes()
    ).digest()
generated = [str(path.relative_to(root)) for path in (root / "artifacts").rglob("*") if path.is_file()]
ignored = subprocess.run(["git", "check-ignore", "--stdin"], input="\n".join(generated),
                         text=True, capture_output=True, check=True).stdout.splitlines()
assert len(ignored) == len(generated)
source_files = subprocess.run(["git", "ls-files", "--others", "--exclude-standard"],
                              text=True, capture_output=True, check=True).stdout.splitlines()
for relative in source_files:
    content = (root / relative).read_text(encoding="utf-8")
    assert not re.search(
        r"(?:(?<![A-Za-z])[A-Za-z]:[/\\]|/(?:home|Users|mnt)/)[^\n]*neoversity-ml-final",
        content,
    ), relative
    assert not any(line.rstrip() != line for line in content.splitlines()), relative
assert not any(path.startswith(("artifacts/", ".venv/", ".tmp/", ".uv-cache/")) for path in source_files)
print(json.dumps({
    "bundle_read_without_mlflow_import": True,
    "oof_rows": 60, "test_rows": 12,
    "mlflow_status": status, "matching_copied_artifacts": len(files),
    "ignored_generated_files": len(generated),
    "reviewable_source_files": len(source_files),
    "old_absolute_paths": 0, "trailing_whitespace": 0,
}, indent=2))
