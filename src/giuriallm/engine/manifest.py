"""Run manifest.

Everything needed to re-run a study and to check a reported number against the
run that produced it: the exact config, its hash, the tool and environment
versions, per-judgement records including failures, and the cost and estimated
energy of the run.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .. import __version__
from ..spec import RunConfig, config_hash, dump_config
from .cost import DEFAULT_GRID_INTENSITY_G_PER_KWH, co2_grams
from .runner import RunResult

MANIFEST_NAME = "manifest.json"
JUDGEMENTS_NAME = "judgements.jsonl"
CONFIG_NAME = "config.yaml"


def build_manifest(config: RunConfig, result: RunResult) -> dict[str, Any]:
    from ..report.summary import report_config

    usage = result.usage
    return {
        "tool": "giuriallm",
        "version": __version__,
        "run_id": result.run_id,
        "config_hash": config_hash(config),
        "dataset_hash": (
            hashlib.sha256(config.dataset.path.read_bytes()).hexdigest()
            if config.dataset.path.is_file()
            else None
        ),
        "human_labels_hash": (
            hashlib.sha256(config.meta_evaluation.human_labels.read_bytes()).hexdigest()
            if config.meta_evaluation and config.meta_evaluation.human_labels.is_file()
            else None
        ),
        "name": config.name,
        "report_config": report_config(config),
        "started_at": result.started_at,
        "finished_at": result.finished_at,
        "written_at": datetime.now(UTC).isoformat(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "judges": [
            {
                "provider": m.provider,
                "id": m.id,
                "samples": m.samples,
                "temperature": m.temperature,
                "thinking": m.thinking.value,
                "effort": m.effort,
            }
            for m in config.judges()
        ],
        "counts": {
            "items": result.n_items,
            "judgements": result.n_judgements,
            "failed_judgements": result.n_failed,
            "cache_hits": result.cache_hits,
            "cache_misses": result.cache_misses,
        },
        "usage": {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
            "cost_usd": round(usage.cost_usd, 6),
            "energy_wh_estimate": round(usage.energy_wh, 4),
            "co2_g_estimate": round(co2_grams(usage.energy_wh), 4),
            "energy_wh_per_ktok": config.engine.energy_wh_per_ktok,
            "grid_intensity_g_per_kwh": DEFAULT_GRID_INTENSITY_G_PER_KWH,
            "energy_is_estimated": True,
        },
        "diagnostics": {
            "flip_rate": result.flip_rate,
            "position_bias": result.position_bias,
            "bradley_terry": result.bradley_terry,
        },
        "meta_evaluation": (
            result.agreement.model_dump(mode="json") if result.agreement is not None else None
        ),
        "notes": result.notes,
        "evaluations": {
            name: value.model_dump(mode="json") for name, value in result.evaluations.items()
        },
        "human_baseline": result.human_baseline,
    }


def write_run(config: RunConfig, result: RunResult, output_dir: Path | str) -> Path:
    """Write the manifest, the resolved config, and every judgement to disk."""
    run_dir = Path(output_dir) / result.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / CONFIG_NAME).write_text(dump_config(config))
    (run_dir / MANIFEST_NAME).write_text(
        json.dumps(build_manifest(config, result), indent=2, default=str) + "\n"
    )

    with (run_dir / JUDGEMENTS_NAME).open("w") as fh:
        for item in result.results:
            for judgement in item.judgements:
                fh.write(judgement.model_dump_json() + "\n")

    return run_dir


def read_manifest(run_dir: Path | str) -> dict[str, Any]:
    return json.loads((Path(run_dir) / MANIFEST_NAME).read_text())
