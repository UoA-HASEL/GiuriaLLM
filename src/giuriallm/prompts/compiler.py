"""Prompt compiler.

Turns a (JudgeSpec, PromptSpec, item, variant) tuple into a concrete request:
a system prompt, a user prompt, and a JSON schema for the verdict. The schema
is enforced through forced tool use, so a verdict is parsed from structured
output rather than from prose.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..spec import (
    Criterion,
    EvaluationType,
    JudgeSpec,
    PromptOrder,
    PromptSpec,
    PromptStrategy,
    ReferenceMode,
    ResultKind,
    ScaleKind,
)
from ..types import EvaluationItem, Variant

TOOL_NAME = "submit_verdict"

LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class CompiledPrompt:
    system: str
    user: str
    schema: dict[str, Any]
    labels: list[str]
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def hash(self) -> str:
        payload = json.dumps(
            {"system": self.system, "user": self.user, "schema": self.schema},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:32]


class PromptCompiler:
    def __init__(self, judge: JudgeSpec, prompt: PromptSpec):
        self.judge = judge
        self.prompt = prompt
        self._examples = _load_examples(prompt.examples)

    # -- schema ----------------------------------------------------------

    def schema(self, n_candidates: int) -> dict[str, Any]:
        labels = LABELS[:n_candidates]
        properties: dict[str, Any] = {}
        required: list[str] = []

        def add_explanation() -> None:
            if self.judge.outputs.explanation:
                properties["explanation"] = {
                    "type": "string",
                    "description": "Reasoning that justifies the verdict, grounded in the "
                    "candidate text and the stated criteria.",
                }
                required.append("explanation")

        def add_result() -> None:
            if self.judge.outputs.result is ResultKind.GRADED:
                properties["scores"] = {
                    "type": "array",
                    "description": "One entry per criterion, in the order listed.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "criterion": {
                                "type": "string",
                                "enum": [c.name for c in self.judge.criteria],
                            },
                            "score": _score_schema(self.judge.criteria),
                        },
                        "required": ["criterion", "score"],
                        "additionalProperties": False,
                    },
                    "minItems": len(self.judge.criteria),
                    "maxItems": len(self.judge.criteria),
                }
                if self.judge.outputs.explanation:
                    entry = properties["scores"]["items"]
                    entry["properties"]["justification"] = {"type": "string"}
                    entry["required"].append("justification")
                required.append("scores")
            elif self.judge.outputs.result is ResultKind.BEST_CHOICE:
                properties["best_choice"] = {
                    "type": "string",
                    "enum": list(labels),
                    "description": "Label of the candidate that best satisfies the criteria.",
                }
                required.append("best_choice")
            else:  # ranking
                properties["ranking"] = {
                    "type": "array",
                    "description": "All candidate labels, best first, each exactly once.",
                    "items": {"type": "string", "enum": list(labels)},
                    "minItems": n_candidates,
                    "maxItems": n_candidates,
                }
                required.append("ranking")

        if self.prompt.order is PromptOrder.EXPLAIN_THEN_RATE:
            add_explanation()
            add_result()
        else:
            add_result()
            add_explanation()

        if self.judge.outputs.feedback:
            properties["feedback"] = {
                "type": "string",
                "description": "Concrete, actionable suggestions for improving the candidate.",
            }
            required.append("feedback")

        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    # -- prompts ---------------------------------------------------------

    def compile(
        self,
        item: EvaluationItem,
        variant: Variant,
        evidence: dict[str, Any] | None = None,
    ) -> CompiledPrompt:
        shown = [item.candidates[i] for i in variant.order]
        labels = list(LABELS[: len(shown)])
        return CompiledPrompt(
            system=self._system(),
            user=self._user(item, shown, labels, evidence or {}),
            schema=self.schema(len(shown)),
            labels=labels,
            evidence=evidence or {},
        )

    def _system(self) -> str:
        parts: list[str] = []
        if self.prompt.system_preamble:
            parts.append(self.prompt.system_preamble.strip())
        else:
            parts.append(
                "You are an expert software engineer acting as an impartial evaluator of "
                "software artifacts."
            )
        parts.append(_task_sentence(self.judge.type, self.judge.outputs.result))
        parts.append(self._criteria_block())

        if self.prompt.strategy is PromptStrategy.COT:
            parts.append("Work through the criteria one at a time before committing to a verdict.")
        elif self.prompt.strategy is PromptStrategy.CRITERIA_DECOMPOSED:
            parts.append(
                "Assess each criterion independently. Do not let a judgement on one "
                "criterion carry over to another."
            )
        elif self.prompt.strategy is PromptStrategy.RUBRIC:
            parts.append(
                "Apply the scale anchors literally. A score is justified only if the "
                "candidate matches that anchor's description."
            )

        if self.judge.type is not EvaluationType.POINTWISE:
            parts.append(
                "Candidate order carries no information: it is randomised. Judge on content "
                "alone, and do not prefer a candidate for being longer or more verbose."
            )

        parts.append(
            "Return only the requested structured verdict, using the supplied output schema. "
            "Do not add text outside the structured output."
        )
        return "\n\n".join(p for p in parts if p)

    def _criteria_block(self) -> str:
        lines = ["Evaluation criteria:"]
        for c in self.judge.criteria:
            lines.append(f"- {c.name}: {c.definition.strip()}")
            lines.append(f"  scale: {_describe_scale(c)}")
            for key in sorted(c.scale.anchors, key=_anchor_sort_key):
                lines.append(f"    {key} = {c.scale.anchors[key]}")
        return "\n".join(lines)

    def _user(
        self,
        item: EvaluationItem,
        shown: list[str],
        labels: list[str],
        evidence: dict[str, Any],
    ) -> str:
        parts: list[str] = []

        if self._examples:
            parts.append(_render_examples(self._examples))

        if item.context:
            parts.append(f"## Task\n{item.context.strip()}")

        if item.reference and self.judge.reference.mode is not ReferenceMode.NONE:
            parts.append(f"## Reference solution\n{item.reference.strip()}")

        if len(shown) == 1:
            parts.append(f"## Candidate\n{shown[0].strip()}")
        else:
            for label, text in zip(labels, shown, strict=True):
                parts.append(f"## Candidate {label}\n{text.strip()}")

        if evidence:
            parts.append(_render_evidence(evidence))

        parts.append(_closing_instruction(self.judge, labels))
        return "\n\n".join(parts)


# -- helpers -------------------------------------------------------------


def _anchor_sort_key(key: str) -> tuple[int, float | str]:
    try:
        return (0, float(key))
    except ValueError:
        return (1, key)


def _score_schema(criteria: list[Criterion]) -> dict[str, Any]:
    kinds = {c.scale.kind for c in criteria}
    if kinds == {ScaleKind.CATEGORICAL}:
        allowed = sorted({v for c in criteria for v in (c.scale.categories or [])})
        return {"type": "string", "enum": allowed}
    if ScaleKind.CATEGORICAL in kinds:
        # Mixed numeric and categorical criteria: accept either, validate per criterion.
        return {"type": ["number", "string"]}
    if kinds <= {ScaleKind.LIKERT, ScaleKind.BINARY}:
        return {"type": "integer"}
    return {"type": "number"}


def _describe_scale(c: Criterion) -> str:
    s = c.scale
    if s.kind is ScaleKind.LIKERT:
        return f"integer from {int(s.min or 0)} to {int(s.max or 0)}, higher is better"
    if s.kind is ScaleKind.CONTINUOUS:
        return f"number from {s.min} to {s.max}, higher is better"
    if s.kind is ScaleKind.BINARY:
        return "1 if the criterion is satisfied, 0 otherwise"
    return "one of: " + ", ".join(s.categories or [])


def _task_sentence(t: EvaluationType, r: ResultKind) -> str:
    if t is EvaluationType.POINTWISE:
        return "Score a single candidate against the criteria below."
    if t is EvaluationType.PAIRWISE:
        if r is ResultKind.BEST_CHOICE:
            return "Compare two candidates and select the better one against the criteria below."
        return "Score each of two candidates against the criteria below."
    return "Rank the candidates from best to worst against the criteria below."


def _closing_instruction(judge: JudgeSpec, labels: list[str]) -> str:
    if judge.outputs.result is ResultKind.BEST_CHOICE:
        return f"Select the better candidate ({' or '.join(labels)}) and report it."
    if judge.outputs.result is ResultKind.RANKING:
        return f"Rank all candidates ({', '.join(labels)}), best first, and report the ranking."
    names = ", ".join(c.name for c in judge.criteria)
    return f"Score the candidate on each criterion ({names}) and report the scores."


def _render_evidence(evidence: dict[str, Any]) -> str:
    lines = ["## Evidence", "Results gathered by running tools on the candidates:"]
    for name, payload in evidence.items():
        lines.append(f"### {name}")
        if isinstance(payload, str):
            lines.append(payload.strip())
        else:
            lines.append(json.dumps(payload, indent=2, default=str))
    return "\n".join(lines)


def _load_examples(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    raw = yaml.safe_load(Path(path).read_text())
    if isinstance(raw, dict) and "examples" in raw:
        raw = raw["examples"]
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}: expected a non-empty list of few-shot examples")
    return raw


def _render_examples(examples: list[dict[str, Any]]) -> str:
    lines = ["## Worked examples"]
    for i, ex in enumerate(examples, start=1):
        lines.append(f"### Example {i}")
        for key, value in ex.items():
            lines.append(f"{key}: {value}")
    return "\n".join(lines)
