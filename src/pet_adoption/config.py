"""Small YAML configuration helpers; no competition schema is assumed."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
        raise ValueError("Configuration must be a mapping with string keys.")
    return payload


def flatten_config(config: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten mappings with dot separators; reject ambiguous dotted keys."""
    flattened: dict[str, Any] = {}
    for key, value in config.items():
        if not isinstance(key, str) or not key or "." in key:
            raise ValueError("Configuration keys must be nonempty strings without dots.")
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping) and value:
            flattened.update(flatten_config(value, name))
        else:
            flattened[name] = value
    return flattened
