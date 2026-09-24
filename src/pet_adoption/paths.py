"""Resolve portable configuration paths relative to an explicit project root."""

from pathlib import Path, PureWindowsPath


def artifact_path(project_root: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    windows_path = PureWindowsPath(relative_path)
    if path.is_absolute() or windows_path.drive or windows_path.root or ".." in path.parts:
        raise ValueError("Artifact paths must be relative and cannot contain traversal.")
    root = (project_root / "artifacts").resolve()
    resolved = (project_root / path).resolve()
    if not resolved.is_relative_to(root) or resolved == root:
        raise ValueError("Artifact paths must be inside artifacts/.")
    return resolved


def sqlite_uri(database: Path) -> str:
    """SQLAlchemy accepts an absolute POSIX-form path on Windows and Linux."""
    return f"sqlite:///{database.resolve().as_posix()}"
