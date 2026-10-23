"""Runtime data types: the evaluation item X, the verdict (Y, E, F), and the
per-call judgement record that carries them plus provenance.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EvaluationItem(BaseModel):
    """X: the content being judged, plus its context and optional reference R."""

    id: str
    context: str | None = None
    candidates: list[str] = Field(min_length=1)
    reference: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class CriterionScore(BaseModel):
    criterion: str
    score: float | str
    justification: str | None = None


class Verdict(BaseModel):
    """Y, E, F for one judged item under one prompt variant."""

    scores: list[CriterionScore] = Field(default_factory=list)
    ranking: list[int] | None = None
    best_choice: int | None = None
    explanation: str | None = None
    feedback: str | None = None

    def score_for(self, criterion: str) -> float | str | None:
        for s in self.scores:
            if s.criterion == criterion:
                return s.score
        return None


class Status(str, Enum):
    OK = "ok"
    PARSE_ERROR = "parse_error"
    PROVIDER_ERROR = "provider_error"


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    energy_wh: float = 0.0
    latency_s: float = 0.0
    cached: bool = False

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            energy_wh=self.energy_wh + other.energy_wh,
            latency_s=self.latency_s + other.latency_s,
            cached=self.cached and other.cached,
        )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class Variant(BaseModel):
    """A single debiasing view of an item: which candidate order was shown."""

    order: list[int]
    """Permutation applied to the candidate list before prompting."""

    index: int = 0
    """Index of this variant within the item's variant set."""

    @property
    def is_canonical(self) -> bool:
        return self.order == sorted(self.order)

    def invert(self, position: int) -> int:
        """Map a position in the shown order back to the original candidate index."""
        return self.order[position]


class Judgement(BaseModel):
    """One LLM call's worth of evidence: prompt variant, verdict, provenance."""

    item_id: str
    model: str
    sample: int
    variant: Variant
    status: Status = Status.OK
    verdict: Verdict | None = None
    error: str | None = None
    raw_output: str | None = None
    usage: Usage = Field(default_factory=Usage)
    evidence: dict[str, Any] = Field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status is Status.OK and self.verdict is not None


class ItemResult(BaseModel):
    """Aggregated outcome for one item, plus the judgements it was derived from."""

    item_id: str
    aggregate: Verdict | None = None
    selected_candidate: int | None = None
    """Candidate selected by voting, separate from a complete ranking."""
    order_choices: dict[int, int] = Field(default_factory=dict)
    """Selected original candidate index for each valid presentation order."""
    judgements: list[Judgement] = Field(default_factory=list)
    flipped: bool | None = None
    """Pairwise/listwise only: did the verdict change when the order changed?"""
    disagreement: float | None = None
    """Spread across judgements (std-dev for graded, 1-consensus for choices)."""

    @property
    def usage(self) -> Usage:
        total = Usage()
        for j in self.judgements:
            total = total + j.usage
        return total
