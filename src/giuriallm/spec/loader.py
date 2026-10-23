"""Loading, hashing, and round-tripping run configurations.

The config is the contract: a run is reproducible from (config hash, dataset).
The hash is computed over the canonical JSON dump of the validated model, so
key order and YAML formatting do not affect it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .models import RunConfig


class ConfigError(ValueError):
    """A configuration failed to load or validate, with an actionable message."""


def _format_validation_error(path: Path | None, err: ValidationError) -> str:
    where = f"{path}" if path else "<config>"
    lines = [f"invalid configuration ({where}):"]
    for e in err.errors():
        loc = ".".join(str(p) for p in e["loc"]) or "<root>"
        lines.append(f"  {loc}: {e['msg']}")
    return "\n".join(lines)


def load_config(path: str | Path) -> RunConfig:
    """Load and validate a YAML config. Raises ConfigError with a usable message."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level of a config must be a mapping")
    config = parse_config(raw, source=path)
    return _resolve_paths(config, base=path.parent)


def parse_config(raw: dict[str, Any], source: Path | None = None) -> RunConfig:
    try:
        return RunConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(source, exc)) from exc


def _resolve_paths(config: RunConfig, base: Path) -> RunConfig:
    """Make dataset/label/example paths relative to the config file, not the cwd."""
    data = config.model_dump(mode="python")
    data["dataset"]["path"] = _abs(base, config.dataset.path)
    if config.meta_evaluation is not None:
        data["meta_evaluation"]["human_labels"] = _abs(base, config.meta_evaluation.human_labels)
    if config.prompt.examples is not None:
        data["prompt"]["examples"] = _abs(base, config.prompt.examples)
    return RunConfig.model_validate(data)


def _abs(base: Path, p: Path) -> Path:
    return p if p.is_absolute() else (base / p).resolve()


def canonical_json(config: RunConfig) -> str:
    """Canonical serialisation used for hashing: sorted keys, no whitespace noise."""
    payload = json.loads(config.model_dump_json(exclude_none=False))
    # Volatile, non-semantic fields must not change the hash.
    for key in ("engine", "report"):
        payload.pop(key, None)
    payload.pop("name", None)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(config: RunConfig) -> str:
    """Stable hash of the semantic content of a config (16 hex chars)."""
    return hashlib.sha256(canonical_json(config).encode()).hexdigest()[:16]


def dump_config(config: RunConfig) -> str:
    """Round-trippable YAML for the manifest."""
    payload = json.loads(config.model_dump_json())
    return yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
