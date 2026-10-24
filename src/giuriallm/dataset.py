"""Dataset loading: JSONL rows -> EvaluationItem, under the field mapping in the
config. A row that cannot be mapped is an error, never a silently dropped item.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .spec import DatasetSpec, ReferenceMode
from .types import EvaluationItem


class DatasetError(ValueError):
    pass


@dataclass
class HumanAnnotations:
    """Labels and optional, position-aligned annotator identities for one item."""

    labels: list[Any]
    annotators: list[str] | None = None


def _get(row: dict[str, Any], field: str, line: int, path: Path) -> Any:
    if field not in row:
        raise DatasetError(
            f"{path}:{line}: row has no field {field!r}; "
            f"available fields: {sorted(row)}. Fix dataset.fields in the config."
        )
    return row[field]


def load_items(
    spec: DatasetSpec, reference_mode: ReferenceMode = ReferenceMode.NONE
) -> list[EvaluationItem]:
    """Read a JSONL dataset into evaluation items."""
    path = Path(spec.path)
    if not path.is_file():
        raise DatasetError(f"dataset not found: {path}")

    items: list[EvaluationItem] = []
    seen: set[str] = set()
    for line, row in _rows(path):
        item_id = str(row[spec.fields.id]) if spec.fields.id in row else f"item-{line}"
        if item_id in seen:
            raise DatasetError(f"{path}:{line}: duplicate item id {item_id!r}")
        seen.add(item_id)

        context = None
        if spec.fields.context is not None:
            context = str(_get(row, spec.fields.context, line, path))

        candidates = [str(_get(row, f, line, path)) for f in spec.fields.candidates]

        reference = None
        if spec.fields.reference is not None:
            if spec.fields.reference in row:
                reference = str(row[spec.fields.reference])
            elif reference_mode is ReferenceMode.REQUIRED:
                raise DatasetError(
                    f"{path}:{line}: judge.reference.mode=required but the row has no "
                    f"{spec.fields.reference!r} field"
                )

        consumed = {spec.fields.id, spec.fields.context, spec.fields.reference}
        consumed.update(spec.fields.candidates)
        meta = {k: v for k, v in row.items() if k not in consumed}

        items.append(
            EvaluationItem(
                id=item_id,
                context=context,
                candidates=candidates,
                reference=reference,
                meta=meta,
            )
        )
        if spec.limit is not None and len(items) >= spec.limit:
            break

    if not items:
        raise DatasetError(f"{path}: dataset is empty")
    return items


def _rows(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open() as fh:
        for line, text in enumerate(fh, start=1):
            text = text.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path}:{line}: not valid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise DatasetError(
                    f"{path}:{line}: expected a JSON object, got {type(row).__name__}"
                )
            yield line, row


def load_human_labels(path: Path) -> dict[str, list[Any]]:
    """Read human labels for meta-evaluation.

    Each row is ``{"id": ..., "label": <value>}`` or ``{"id": ..., "labels": [...]}``.
    The list form carries the *distribution* of expert opinion for one item, which
    the distribution-aware metrics consume directly.
    """
    return {key: row.labels for key, row in load_human_annotations(path).items()}


def load_human_annotations(path: Path) -> dict[str, HumanAnnotations]:
    """Load labels, optionally carrying a parallel ``annotators`` array.

    Annotator IDs are never inferred from array positions. Repeated rows for an
    item are concatenated, as in :func:`load_human_labels`; repeated annotator
    identities within an item are rejected rather than counted twice.
    """
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"human labels not found: {path}")
    labels: dict[str, HumanAnnotations] = {}
    for line, row in _rows(path):
        if "id" not in row:
            raise DatasetError(f"{path}:{line}: human label row has no 'id' field")
        key = str(row["id"])
        if "labels" in row:
            values = row["labels"]
            if not isinstance(values, list) or not values:
                raise DatasetError(f"{path}:{line}: 'labels' must be a non-empty list")
        elif "label" in row:
            values = [row["label"]]
        else:
            raise DatasetError(f"{path}:{line}: expected a 'label' or 'labels' field")
        annotators = row.get("annotators")
        if annotators is not None:
            if (
                not isinstance(annotators, list)
                or len(annotators) != len(values)
                or any(not isinstance(a, str) or not a.strip() for a in annotators)
            ):
                raise DatasetError(
                    f"{path}:{line}: 'annotators' must contain one non-empty string ID per label"
                )
            if len(set(annotators)) != len(annotators):
                raise DatasetError(f"{path}:{line}: duplicate annotator ID for item {key!r}")
        if key not in labels:
            labels[key] = HumanAnnotations(list(values), annotators)
            continue
        previous = labels[key]
        if (previous.annotators is None) != (annotators is None):
            raise DatasetError(
                f"{path}:{line}: item {key!r} mixes labels with and without annotator IDs"
            )
        if annotators is not None:
            assert previous.annotators is not None
            if set(previous.annotators) & set(annotators):
                raise DatasetError(f"{path}:{line}: duplicate annotator ID for item {key!r}")
            previous.annotators.extend(annotators)
        previous.labels.extend(values)
    if not labels:
        raise DatasetError(f"{path}: no human labels found")
    return labels
