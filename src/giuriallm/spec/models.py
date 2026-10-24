"""Configuration schema.

This module is a direct encoding of the LLM-as-a-Judge formalism

    E(T, C, X, R) -> (Y, E, F)

from He et al., *From Code to Courtroom* (papers/2510.24367v1.pdf, Sec. 2).
Every key here traces back to a symbol in that tuple; see AGENTS.md Sec. 1.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Strict(BaseModel):
    """Base model: unknown keys are an error, never a silent default."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------
# T -- evaluation type
# --------------------------------------------------------------------------


class EvaluationType(str, Enum):
    POINTWISE = "pointwise"
    PAIRWISE = "pairwise"
    LISTWISE = "listwise"


# --------------------------------------------------------------------------
# C -- evaluation criteria
# --------------------------------------------------------------------------


class ScaleKind(str, Enum):
    LIKERT = "likert"
    CONTINUOUS = "continuous"
    BINARY = "binary"
    CATEGORICAL = "categorical"


class Scale(Strict):
    kind: ScaleKind = ScaleKind.LIKERT
    min: float | None = None
    max: float | None = None
    categories: list[str] | None = None
    anchors: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> Scale:
        if self.kind in (ScaleKind.LIKERT, ScaleKind.CONTINUOUS):
            if self.min is None or self.max is None:
                raise ValueError(f"scale.kind={self.kind.value} requires 'min' and 'max'")
            if self.min >= self.max:
                raise ValueError("scale.min must be < scale.max")
        if self.kind is ScaleKind.CATEGORICAL and not self.categories:
            raise ValueError("scale.kind=categorical requires a non-empty 'categories' list")
        for key in self.anchors:
            if self.kind in (ScaleKind.LIKERT, ScaleKind.CONTINUOUS):
                try:
                    float(key)
                except ValueError:
                    raise ValueError(
                        f"anchor key {key!r} is not a number; numeric scales need numeric anchors"
                    ) from None
        return self

    def values(self) -> list[float] | list[str] | None:
        """The discrete support of the scale, when it has one."""
        if self.kind is ScaleKind.LIKERT:
            assert self.min is not None and self.max is not None
            return [float(v) for v in range(int(self.min), int(self.max) + 1)]
        if self.kind is ScaleKind.BINARY:
            return [0.0, 1.0]
        if self.kind is ScaleKind.CATEGORICAL:
            return list(self.categories or [])
        return None


class Criterion(Strict):
    name: str = Field(min_length=1)
    definition: str = Field(min_length=1)
    scale: Scale = Field(default_factory=lambda: Scale(kind=ScaleKind.LIKERT, min=1, max=5))
    weight: float = 1.0


# --------------------------------------------------------------------------
# R -- reference
# --------------------------------------------------------------------------


class ReferenceMode(str, Enum):
    NONE = "none"
    OPTIONAL = "optional"
    REQUIRED = "required"


class Reference(Strict):
    mode: ReferenceMode = ReferenceMode.NONE


# --------------------------------------------------------------------------
# Y, E, F -- outputs
# --------------------------------------------------------------------------


class ResultKind(str, Enum):
    GRADED = "graded"
    RANKING = "ranking"
    BEST_CHOICE = "best_choice"


class Outputs(Strict):
    result: ResultKind = ResultKind.GRADED
    explanation: bool = True
    feedback: bool = False


class JudgeSpec(Strict):
    type: EvaluationType
    criteria: list[Criterion] = Field(min_length=1)
    reference: Reference = Field(default_factory=Reference)
    outputs: Outputs = Field(default_factory=Outputs)

    @model_validator(mode="after")
    def _check_type_result_compatible(self) -> JudgeSpec:
        t, r = self.type, self.outputs.result
        allowed: dict[EvaluationType, set[ResultKind]] = {
            EvaluationType.POINTWISE: {ResultKind.GRADED},
            # Pairwise returns a choice: a single graded verdict cannot say which
            # of the two candidates a score belongs to. Score two candidates by
            # running a pointwise evaluation over each.
            EvaluationType.PAIRWISE: {ResultKind.BEST_CHOICE},
            EvaluationType.LISTWISE: {ResultKind.RANKING, ResultKind.BEST_CHOICE},
        }
        if r not in allowed[t]:
            raise ValueError(
                f"judge.outputs.result={r.value} is not valid for judge.type={t.value}; "
                f"expected one of {sorted(k.value for k in allowed[t])}"
            )
        names = [c.name for c in self.criteria]
        if len(names) != len(set(names)):
            raise ValueError("judge.criteria names must be unique")
        return self


# --------------------------------------------------------------------------
# Model / provider
# --------------------------------------------------------------------------


class ThinkingMode(str, Enum):
    DISABLED = "disabled"
    ADAPTIVE = "adaptive"


class ModelSpec(Strict):
    provider: str = "anthropic"
    id: str = "claude-opus-5"
    temperature: float | None = 0.0
    max_tokens: int = 4096
    samples: int = Field(default=1, ge=1)
    thinking: ThinkingMode = ThinkingMode.DISABLED
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    price_per_mtok: tuple[float, float] | None = None
    label: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    """Provider-specific settings: ``base_url`` for OpenAI-compatible servers and
    Ollama, ``structured_output`` mode, ``timeout``, ``keep_alive``, ``num_ctx``.
    Kept in one bag so the T/C/X/R -> Y/E/F keys stay provider-neutral, and
    included in the config hash because they change what a run does."""

    @property
    def display(self) -> str:
        return self.label or f"{self.provider}:{self.id}"


# --------------------------------------------------------------------------
# Prompt compilation
# --------------------------------------------------------------------------


class PromptStrategy(str, Enum):
    DIRECT = "direct"
    COT = "cot"
    RUBRIC = "rubric"
    FEWSHOT = "fewshot"
    CRITERIA_DECOMPOSED = "criteria_decomposed"


class PromptOrder(str, Enum):
    EXPLAIN_THEN_RATE = "explain_then_rate"
    RATE_THEN_EXPLAIN = "rate_then_explain"


class PromptSpec(Strict):
    strategy: PromptStrategy = PromptStrategy.COT
    order: PromptOrder = PromptOrder.EXPLAIN_THEN_RATE
    examples: Path | None = None
    system_preamble: str | None = None

    @model_validator(mode="after")
    def _check_examples(self) -> PromptSpec:
        if self.strategy is PromptStrategy.FEWSHOT and self.examples is None:
            raise ValueError("prompt.strategy=fewshot requires 'prompt.examples' (a YAML file)")
        return self


# --------------------------------------------------------------------------
# Debias
# --------------------------------------------------------------------------


class DebiasSpec(Strict):
    position_swap: bool | None = None
    """Judge (a, b) and (b, a). ``None`` means automatic: on for pairwise, off
    elsewhere. Set ``false`` to opt out; setting ``true`` for a non-pairwise
    evaluation is a configuration error rather than a silently ignored key."""
    permutations: int = Field(default=1, ge=1)
    report_flip_rate: bool = True
    verbosity_control: bool = False
    self_preference_probe: bool = False


# --------------------------------------------------------------------------
# Panel / aggregation
# --------------------------------------------------------------------------


class Aggregation(str, Enum):
    MAJORITY_VOTE = "majority_vote"
    MEAN = "mean"
    MEDIAN = "median"
    WEIGHTED = "weighted"
    BRADLEY_TERRY = "bradley_terry"
    DAWID_SKENE = "dawid_skene"


class PanelSpec(Strict):
    members: list[ModelSpec] = Field(default_factory=list)
    aggregation: Aggregation = Aggregation.MAJORITY_VOTE
    weights: dict[str, float] = Field(default_factory=dict)


# --------------------------------------------------------------------------
# Evidence (tool-augmented judging)
# --------------------------------------------------------------------------


class EvidencePlugin(Strict):
    name: str
    options: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------


class DatasetFields(Strict):
    id: str = "id"
    context: str | None = "prompt"
    candidates: list[str] = Field(default_factory=lambda: ["candidate"])
    reference: str | None = None


class DatasetSpec(Strict):
    path: Path
    fields: DatasetFields = Field(default_factory=DatasetFields)
    limit: int | None = None


# --------------------------------------------------------------------------
# Meta-evaluation
# --------------------------------------------------------------------------


class Metric(str, Enum):
    ERROR = "error"
    ACCURACY = "accuracy"
    COHEN_KAPPA = "cohen_kappa"
    KRIPPENDORFF_ALPHA = "krippendorff_alpha"
    SPEARMAN = "spearman"
    KENDALL = "kendall"
    ICC = "icc"
    EARTH_MOVERS_DISTANCE = "earth_movers_distance"
    JENSEN_SHANNON = "jensen_shannon"


class MetaEvaluationSpec(Strict):
    human_labels: Path
    metrics: list[Metric] = Field(default_factory=list)
    stratify_by_agreement: bool = True
    agreement_threshold: float = Field(default=0.8, gt=0.0, le=1.0)


# --------------------------------------------------------------------------
# Engine / reporting
# --------------------------------------------------------------------------


class EngineSpec(Strict):
    concurrency: int = Field(default=8, ge=1)
    max_retries: int = Field(default=3, ge=0)
    cache: bool = True
    cache_dir: Path = Path(".giuriallm-cache")
    energy_wh_per_ktok: float = 0.4
    """Estimated inference energy per 1k tokens, in watt-hours.

    A configurable coefficient, not a measurement: hosted-API energy use is not
    observable from the client. Reported as an estimate and recorded in the
    manifest so a reader can substitute their own figure.
    """


class ReportSpec(Strict):
    output_dir: Path = Path("runs")
    formats: list[Literal["markdown", "html", "json", "latex"]] = Field(
        default_factory=lambda: ["markdown", "json"]
    )


# --------------------------------------------------------------------------
# Top-level run configuration
# --------------------------------------------------------------------------


class RunConfig(Strict):
    judge: JudgeSpec
    model: ModelSpec = Field(default_factory=ModelSpec)
    prompt: PromptSpec = Field(default_factory=PromptSpec)
    debias: DebiasSpec = Field(default_factory=DebiasSpec)
    panel: PanelSpec | None = None
    evidence: list[EvidencePlugin] = Field(default_factory=list)
    dataset: DatasetSpec
    meta_evaluation: MetaEvaluationSpec | None = None
    engine: EngineSpec = Field(default_factory=EngineSpec)
    report: ReportSpec = Field(default_factory=ReportSpec)
    name: str | None = None

    @model_validator(mode="after")
    def _check(self) -> RunConfig:
        n = len(self.dataset.fields.candidates)
        expected: dict[EvaluationType, str] = {
            EvaluationType.POINTWISE: "exactly 1",
            EvaluationType.PAIRWISE: "exactly 2",
            EvaluationType.LISTWISE: "at least 3",
        }
        ok = {
            EvaluationType.POINTWISE: n == 1,
            EvaluationType.PAIRWISE: n == 2,
            EvaluationType.LISTWISE: n >= 3,
        }[self.judge.type]
        if not ok:
            raise ValueError(
                f"judge.type={self.judge.type.value} needs {expected[self.judge.type]} "
                f"candidate field(s) in dataset.fields.candidates; got {n}"
            )
        needs_reference = self.judge.reference.mode is ReferenceMode.REQUIRED
        if needs_reference and not self.dataset.fields.reference:
            raise ValueError(
                "judge.reference.mode=required but dataset.fields.reference is not set"
            )
        if self.debias.position_swap is True and self.judge.type is not EvaluationType.PAIRWISE:
            raise ValueError(
                "debias.position_swap applies to judge.type=pairwise only; "
                "use debias.permutations for listwise"
            )
        if self.debias.permutations > 1 and self.judge.type is not EvaluationType.LISTWISE:
            raise ValueError("debias.permutations > 1 applies to judge.type=listwise only")
        if self.panel is not None and not self.panel.members:
            raise ValueError("panel is present but panel.members is empty; omit 'panel' instead")
        names = [model.display for model in self.judges()]
        if len(set(names)) != len(names) or any(n.lower() in ("panel", "human") for n in names):
            raise ValueError("Judge labels must be unique; 'Panel' and 'Human' are reserved")
        return self

    def judges(self) -> list[ModelSpec]:
        """The models that will produce verdicts: the panel, or the single model."""
        if self.panel is not None:
            return list(self.panel.members)
        return [self.model]


AnyConfig = Annotated[RunConfig, Field()]
