"""Verdict parsing.

A parse failure is recorded as a failure. It is never coerced into a score and
the item is never silently dropped (AGENTS.md Sec. 5).
"""

from __future__ import annotations

from typing import Any

from ..spec import Criterion, JudgeSpec, ResultKind, ScaleKind
from ..types import CriterionScore, Variant, Verdict


class VerdictParseError(ValueError):
    pass


def parse_verdict(
    payload: dict[str, Any],
    judge: JudgeSpec,
    variant: Variant,
    labels: list[str],
) -> Verdict:
    """Map raw structured output onto a Verdict in *original* candidate indices."""
    if not isinstance(payload, dict):
        raise VerdictParseError(f"expected a JSON object, got {type(payload).__name__}")

    verdict = Verdict(
        explanation=_opt_str(payload.get("explanation")),
        feedback=_opt_str(payload.get("feedback")),
    )

    if judge.outputs.explanation and verdict.explanation is None:
        raise VerdictParseError("judge.outputs.explanation is true but no explanation was returned")
    if judge.outputs.feedback and verdict.feedback is None:
        raise VerdictParseError("judge.outputs.feedback is true but no feedback was returned")

    if judge.outputs.result is ResultKind.GRADED:
        verdict.scores = _parse_scores(payload, judge)
    elif judge.outputs.result is ResultKind.BEST_CHOICE:
        verdict.best_choice = variant.invert(_label_index(payload.get("best_choice"), labels))
    else:
        verdict.ranking = _parse_ranking(payload.get("ranking"), labels, variant)

    return verdict


def _parse_scores(payload: dict[str, Any], judge: JudgeSpec) -> list[CriterionScore]:
    raw = payload.get("scores")
    if not isinstance(raw, list):
        raise VerdictParseError("'scores' is missing or not a list")

    by_name = {c.name: c for c in judge.criteria}
    seen: dict[str, CriterionScore] = {}
    for entry in raw:
        if not isinstance(entry, dict):
            raise VerdictParseError(f"score entry is not an object: {entry!r}")
        name = entry.get("criterion")
        if name not in by_name:
            raise VerdictParseError(
                f"unknown criterion {name!r}; expected one of {sorted(by_name)}"
            )
        if name in seen:
            raise VerdictParseError(f"criterion {name!r} scored more than once")
        seen[name] = CriterionScore(
            criterion=name,
            score=_coerce_score(entry.get("score"), by_name[name]),
            justification=_opt_str(entry.get("justification")),
        )

    missing = sorted(set(by_name) - set(seen))
    if missing:
        raise VerdictParseError(f"no score returned for criteria: {missing}")
    return [seen[c.name] for c in judge.criteria]


def _coerce_score(value: Any, criterion: Criterion) -> float | str:
    scale = criterion.scale
    if scale.kind is ScaleKind.CATEGORICAL:
        if not isinstance(value, str) or value not in (scale.categories or []):
            raise VerdictParseError(f"{criterion.name}: {value!r} is not one of {scale.categories}")
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VerdictParseError(f"{criterion.name}: score {value!r} is not a number")
    numeric = float(value)
    if scale.kind is ScaleKind.BINARY:
        if numeric not in (0.0, 1.0):
            raise VerdictParseError(f"{criterion.name}: binary score must be 0 or 1, got {numeric}")
        return numeric
    lo, hi = scale.min, scale.max
    if lo is not None and hi is not None and not (lo <= numeric <= hi):
        raise VerdictParseError(f"{criterion.name}: score {numeric} outside scale [{lo}, {hi}]")
    if scale.kind is ScaleKind.LIKERT and numeric != int(numeric):
        raise VerdictParseError(f"{criterion.name}: likert score must be an integer, got {numeric}")
    return numeric


def _parse_ranking(value: Any, labels: list[str], variant: Variant) -> list[int]:
    if not isinstance(value, list):
        raise VerdictParseError("'ranking' is missing or not a list")
    if len(value) != len(labels):
        raise VerdictParseError(f"ranking has {len(value)} entries, expected {len(labels)}")
    positions = [_label_index(v, labels) for v in value]
    if sorted(positions) != list(range(len(labels))):
        raise VerdictParseError(f"ranking is not a permutation of {labels}: {value!r}")
    return [variant.invert(p) for p in positions]


def _label_index(value: Any, labels: list[str]) -> int:
    if not isinstance(value, str):
        raise VerdictParseError(f"expected a candidate label from {labels}, got {value!r}")
    token = value.strip().upper()
    if token not in labels:
        raise VerdictParseError(f"{value!r} is not one of the candidate labels {labels}")
    return labels.index(token)


def _opt_str(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise VerdictParseError(f"expected a string, got {type(value).__name__}")
    text = value.strip()
    return text or None
