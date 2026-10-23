"""Regenerate reports from saved measurements, without providers or model calls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..spec import load_config
from ..types import ItemResult, Judgement
from .summary import render_summary, report_config


def saved_manifest(run_dir: Path | str) -> dict:
    root = Path(run_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    config_path = root / "config.yaml"
    if not manifest.get("report_config") and config_path.is_file():
        manifest["report_config"] = report_config(load_config(config_path))
    results_path = root / "results.json"
    if results_path.is_file():
        payload = json.loads(results_path.read_text())
        # Reports may have been refreshed while the original manifest stays immutable.
        for key in ("evaluations", "human_baseline", "report_config"):
            if key in payload:
                manifest[key] = payload[key]
    if config_path.is_file():
        # New presentation metadata can be recovered from the saved configuration.
        # Keep recorded metadata where present; do not rewrite the original manifest.
        metadata = manifest.setdefault("report_config", {})
        for key, value in report_config(load_config(config_path)).items():
            metadata.setdefault(key, value)
    return manifest


def regenerate_reports(run_dir: Path | str, *, labels_path: Path | None = None) -> list[Path]:
    """Refresh derived JSON/text/Markdown; leave config, manifest and raw records intact.

    An optional relocated labels file must match the saved hash. It is used only
    to recover unavailable reasons and remove spurious singleton-label groups.
    Saved metric values, verdicts, usage and timing are never recomputed.
    """
    from ..aggregate.evaluation import distribution_view, evaluation_views
    from ..dataset import load_human_annotations
    from ..metaeval import HIGH, LOW, _reasons, build_alignment, human_agreement

    root = Path(run_dir)
    manifest = saved_manifest(root)
    config = load_config(root / "config.yaml")
    payload = json.loads((root / "results.json").read_text())
    for record in payload.get("items", []):
        for field in ("judges", "orders", "disagreement"):
            record.pop(field, None)
    cfg = manifest["report_config"]
    payload.pop("report_config", None)
    for summary in payload.get("evaluations", {}).values():
        summary.pop("usage_per_item", None)

    candidate = labels_path or (
        config.meta_evaluation.human_labels if config.meta_evaluation else None
    )
    if candidate and Path(candidate).is_file():
        expected = manifest.get("human_labels_hash")
        actual = hashlib.sha256(Path(candidate).read_bytes()).hexdigest()
        if expected != actual:
            raise ValueError(
                "Reference labels do not match the saved run; reports were not changed."
            )
        raw = [
            Judgement.model_validate_json(line)
            for line in (root / "judgements.jsonl").read_text().splitlines()
            if line.strip()
        ]
        items = [
            ItemResult.model_validate(
                {**record, "judgements": [j for j in raw if j.item_id == record["item_id"]]}
            )
            for record in payload.get("items", [])
        ]
        annotations = load_human_annotations(Path(candidate))
        labels = {i: a.labels for i, a in annotations.items()}
        all_views = [evaluation_views(config, item) for item in items]
        for name, summary in payload.get("evaluations", {}).items():
            metric_views = [
                distribution_view(
                    v[name],
                    [v[m.display] for m in config.judges()] if name == "panel" else [v[name]],
                )
                for v in all_views
            ]
            alignments = {a.target: a for a in build_alignment(config, metric_views, labels)}
            for target in (summary.get("agreement") or {}).get("targets", []):
                a = alignments[target["target"]]
                ids = a.items()
                target["unavailable"] = _reasons(a, ids, target.get("metrics", {}))
                if not cfg.get("stratify"):
                    continue
                groups = {HIGH: [], LOW: []}
                for item_id in ids:
                    agreement = human_agreement(a, item_id)
                    if agreement is not None:
                        groups[
                            HIGH if agreement >= config.meta_evaluation.agreement_threshold else LOW
                        ].append(item_id)
                target["n_ungrouped"] = len(ids) - sum(map(len, groups.values()))
                if ids and target["n_ungrouped"] == len(ids):
                    target["strata"] = {}
                    target["stratum_sizes"] = {}
                    target["stratum_unavailable"] = {}
                elif target.get("strata") or any(groups.values()):
                    target.setdefault("stratum_unavailable", {})
                    for group, group_ids in groups.items():
                        size = target.get("stratum_sizes", {}).get(group, 0)
                        if size != len(group_ids):
                            raise ValueError(
                                "Grouping membership changed; "
                                "explicit metric recomputation is required."
                            )
                        target.setdefault("stratum_sizes", {})[group] = size
                        values = target.setdefault("strata", {}).setdefault(
                            group, {m: None for m in target.get("metrics", {})}
                        )
                        target["stratum_unavailable"][group] = _reasons(
                            a, group_ids, values, empty="empty_group"
                        )
    manifest["evaluations"] = payload.get("evaluations", {})
    primary = "panel" if config.panel else config.model.display
    if primary in manifest["evaluations"]:
        payload["agreement"] = manifest["evaluations"][primary].get("agreement")
    paths = [root / "report.md", root / "console.txt", root / "results.json"]
    paths[0].write_text(render_summary(manifest, markdown=True), encoding="utf-8")
    paths[1].write_text(render_summary(manifest), encoding="utf-8")
    paths[2].write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return paths
