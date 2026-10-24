"""Meta-evaluation: judging the judge.

Alignment with human labels ships with the framework rather than living in a
per-study script. Two things distinguish what is computed here from the usual
practice:

1. **Distribution-aware metrics.** When human labels are supplied as a
   *distribution* (several experts per item), earth mover's distance and
   Jensen-Shannon divergence compare the judge's distribution of verdicts to the
   experts' distribution, instead of collapsing expert disagreement into a
   single gold label that the classic statistics then presume.
2. **Stratification by human agreement.** Every metric is also reported
   separately over items the experts agreed on and items they did not. A judge
   that matches humans on easy items and diverges on contested ones looks fine
   in a pooled number and is exposed by the split.
"""

from __future__ import annotations

from collections import Counter
from statistics import mean

from pydantic import BaseModel, Field

from ..dataset import HumanAnnotations
from ..spec import Criterion, Metric, ResultKind, RunConfig, ScaleKind
from ..types import ItemResult
from .metrics import (
    accuracy,
    cohen_kappa,
    distribution,
    earth_movers_distance,
    icc,
    jensen_shannon,
    kendall_tau,
    krippendorff_alpha,
    spearman,
)

__all__ = [
    "AgreementReport",
    "Alignment",
    "TargetReport",
    "build_alignment",
    "evaluate_alignment",
    "HumanBaselineReport",
    "build_human_baseline",
]

HIGH = "high_agreement"
LOW = "low_agreement"


class Alignment(BaseModel):
    """Paired judge and human values for one evaluation target (a criterion, or
    the choice itself), keyed by item."""

    target: str
    numeric: bool
    binary: bool = False
    support: list[float] | list[str] | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    judge: dict[str, list[float | str]] = Field(default_factory=dict)
    human: dict[str, list[float | str]] = Field(default_factory=dict)
    points: dict[str, float | str] = Field(default_factory=dict)
    """Final aggregated verdicts for point metrics; distributions stay separate."""

    def items(self) -> list[str]:
        return sorted(set(self.judge) & set(self.human))


class TargetReport(BaseModel):
    target: str
    n_items: int
    metrics: dict[str, float | None] = Field(default_factory=dict)
    strata: dict[str, dict[str, float | None]] = Field(default_factory=dict)
    stratum_sizes: dict[str, int] = Field(default_factory=dict)
    unavailable: dict[str, str] = Field(default_factory=dict)
    stratum_unavailable: dict[str, dict[str, str]] = Field(default_factory=dict)
    n_ungrouped: int = 0


class AgreementReport(BaseModel):
    targets: list[TargetReport] = Field(default_factory=list)
    n_matched: int = 0
    n_unmatched_judge: int = 0
    n_unmatched_human: int = 0
    agreement_threshold: float = 0.8
    notes: list[str] = Field(default_factory=list)

    def flat(self) -> dict[str, float | None]:
        """All metrics as ``target/metric`` keys, for tables and CSV export."""
        out: dict[str, float | None] = {}
        for target in self.targets:
            for name, value in target.metrics.items():
                out[f"{target.target}/{name}"] = value
            for stratum, metrics in target.strata.items():
                for name, value in metrics.items():
                    out[f"{target.target}/{stratum}/{name}"] = value
        return out


class HumanTargetReport(BaseModel):
    target: str
    n_annotators: int = 0
    metrics: dict[str, float | None] = Field(default_factory=dict)
    strata: dict[str, dict[str, float | None]] = Field(default_factory=dict)
    metric_annotators: dict[str, int] = Field(default_factory=dict)
    stratum_annotators: dict[str, dict[str, int]] = Field(default_factory=dict)
    per_annotator: dict[str, TargetReport] = Field(default_factory=dict)
    unavailable: dict[str, str] = Field(default_factory=dict)
    stratum_unavailable: dict[str, dict[str, str]] = Field(default_factory=dict)


class HumanBaselineReport(BaseModel):
    available: bool = False
    n_items: int = 0
    targets: list[HumanTargetReport] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Building alignments from run results
# --------------------------------------------------------------------------


def build_alignment(
    config: RunConfig,
    results: list[ItemResult],
    human_labels: dict[str, list],
) -> list[Alignment]:
    """Pair every judged item with its human labels, one alignment per target."""
    if config.judge.outputs.result is ResultKind.GRADED:
        return [
            _graded_alignment(criterion, results, human_labels)
            for criterion in config.judge.criteria
        ]
    return [_choice_alignment(config, results, human_labels)]


def _graded_alignment(
    criterion: Criterion, results: list[ItemResult], human_labels: dict[str, list]
) -> Alignment:
    numeric = criterion.scale.kind not in (ScaleKind.CATEGORICAL, ScaleKind.BINARY)
    alignment = Alignment(
        target=criterion.name,
        numeric=numeric,
        binary=criterion.scale.kind is ScaleKind.BINARY,
        support=(
            criterion.scale.values() if numeric else [_label(v) for v in criterion.scale.values()]
        ),
        lower_bound=criterion.scale.min,
        upper_bound=criterion.scale.max,
    )
    for result in results:
        values = [
            j.verdict.score_for(criterion.name)
            for j in result.judgements
            if j.ok and j.verdict is not None and j.verdict.score_for(criterion.name) is not None
        ]
        if values:
            alignment.judge[result.item_id] = _normalise(values, numeric)
        point = result.aggregate.score_for(criterion.name) if result.aggregate else None
        if point is not None:
            alignment.points[result.item_id] = float(point) if numeric else _label(point)
    for item_id, labels in human_labels.items():
        alignment.human[item_id] = _normalise(labels, numeric)
    return alignment


def _choice_alignment(
    config: RunConfig, results: list[ItemResult], human_labels: dict[str, list]
) -> Alignment:
    n = len(config.dataset.fields.candidates)
    alignment = Alignment(
        target="choice",
        numeric=False,
        support=[str(i) for i in range(n)],
    )
    for result in results:
        values: list[float | str] = []
        for j in result.judgements:
            if not j.ok or j.verdict is None:
                continue
            if j.verdict.best_choice is not None:
                values.append(str(j.verdict.best_choice))
            elif j.verdict.ranking:
                values.append(str(j.verdict.ranking[0]))
        if values:
            alignment.judge[result.item_id] = values
        point = getattr(result, "selected_candidate", None)
        if point is None and result.aggregate is not None:
            point = result.aggregate.best_choice
            if point is None and result.aggregate.ranking:
                point = result.aggregate.ranking[0]
        if point is not None:
            alignment.points[result.item_id] = _label(point)
    for item_id, labels in human_labels.items():
        alignment.human[item_id] = [_label(v) for v in labels]
    return alignment


# --------------------------------------------------------------------------
# Metric computation
# --------------------------------------------------------------------------


def evaluate_alignment(
    alignments: list[Alignment],
    metrics: list[Metric],
    stratify: bool = True,
    threshold: float = 0.8,
) -> AgreementReport:
    report = AgreementReport(agreement_threshold=threshold)
    if not alignments:
        return report

    first = alignments[0]
    matched = set(first.judge) & set(first.human)
    report.n_matched = len(matched)
    report.n_unmatched_judge = len(set(first.judge) - set(first.human))
    report.n_unmatched_human = len(set(first.human) - set(first.judge))
    if report.n_unmatched_judge:
        report.notes.append(
            f"{report.n_unmatched_judge} judged item(s) have no human label and were excluded"
        )
    if report.n_unmatched_human:
        report.notes.append(
            f"{report.n_unmatched_human} labelled item(s) were not judged and were excluded"
        )

    for alignment in alignments:
        report.targets.append(_target_report(alignment, metrics, stratify, threshold))
    return report


def _target_report(
    alignment: Alignment, metrics: list[Metric], stratify: bool, threshold: float
) -> TargetReport:
    ids = alignment.items()
    report = TargetReport(target=alignment.target, n_items=len(ids))
    report.metrics = _compute(alignment, ids, metrics)
    report.unavailable = _reasons(alignment, ids, report.metrics)

    if stratify and ids:
        agreement = {i: human_agreement(alignment, i) for i in ids}
        high = [i for i in ids if agreement[i] is not None and agreement[i] >= threshold]
        low = [i for i in ids if agreement[i] is not None and agreement[i] < threshold]
        report.n_ungrouped = len(ids) - len(high) - len(low)
        # A single reference label is not evidence of annotator agreement.
        if high or low:
            for group, group_ids in ((HIGH, high), (LOW, low)):
                report.stratum_sizes[group] = len(group_ids)
                report.strata[group] = _compute(alignment, group_ids, metrics)
                report.stratum_unavailable[group] = _reasons(
                    alignment, group_ids, report.strata[group], empty="empty_group"
                )
    return report


def human_agreement(alignment: Alignment, item_id: str) -> float | None:
    """How much the experts agreed on one item, in [0, 1].

    Categorical: the modal label's share. Numeric: 1 - mean absolute deviation
    normalised by half the scale range, floored at 0.
    """
    labels = alignment.human.get(item_id, [])
    if len(labels) < 2:
        return None
    if not alignment.numeric:
        counts = Counter(str(v) for v in labels)
        return counts.most_common(1)[0][1] / len(labels)

    values = [float(v) for v in labels]
    centre = mean(values)
    deviation = mean(abs(v - centre) for v in values)
    span = _span(alignment)
    if span is None or span <= 0:
        return None
    return max(0.0, 1.0 - deviation / (span / 2.0))


def _span(alignment: Alignment) -> float | None:
    if alignment.lower_bound is not None and alignment.upper_bound is not None:
        return alignment.upper_bound - alignment.lower_bound
    if not alignment.support:
        return None
    try:
        numbers = [float(v) for v in alignment.support]
    except (TypeError, ValueError):
        return None
    return max(numbers) - min(numbers)


def _compute(
    alignment: Alignment, ids: list[str], metrics: list[Metric]
) -> dict[str, float | None]:
    if not ids:
        return {metric.value: None for metric in metrics}
    judge_points = [
        alignment.points[i]
        if i in alignment.points
        else _collapse(alignment.judge[i], alignment.numeric, alignment.binary)
        for i in ids
    ]
    human_points = [_collapse(alignment.human[i], alignment.numeric, alignment.binary) for i in ids]

    out: dict[str, float | None] = {}
    for metric in metrics:
        out[metric.value] = _one(metric, alignment, ids, judge_points, human_points)
    return out


def _reasons(alignment, ids, values, *, empty="no_matched_items") -> dict[str, str]:
    """Persist causes at computation time; renderers never guess from n/a."""
    reasons = {}
    for metric, value in values.items():
        if value is not None:
            continue
        reason = "undefined"
        if not ids:
            reason = empty
            if empty != "empty_group":
                if not alignment.judge:
                    reason = "no_valid_judgements"
                elif not alignment.human:
                    reason = "no_reference_labels"
        elif metric in ("spearman", "kendall", "icc"):
            if not alignment.numeric:
                reason = "requires_numeric"
            elif len(ids) < 2:
                reason = "insufficient_pairs"
            elif metric in ("spearman", "kendall"):
                judge = [alignment.points.get(i, _collapse(alignment.judge[i], True)) for i in ids]
                human = [_collapse(alignment.human[i], True) for i in ids]
                if len(set(judge)) < 2 or len(set(human)) < 2:
                    reason = "constant_values"
        elif metric == "cohen_kappa":
            reason = "kappa_undefined"
        elif metric == "error" and alignment.numeric:
            reason = "invalid_scale"
        elif metric in ("earth_movers_distance", "jensen_shannon") and not alignment.support:
            reason = "missing_support"
        reasons[metric] = reason
    return reasons


def _one(
    metric: Metric,
    alignment: Alignment,
    ids: list[str],
    judge_points: list,
    human_points: list,
) -> float | None:
    if metric is Metric.ERROR:
        if not judge_points:
            return None
        if not alignment.numeric:
            return mean(
                float(_label(j) != _label(h))
                for j, h in zip(judge_points, human_points, strict=True)
            )
        span = _span(alignment)
        if span is None or span <= 0:
            return None
        return mean(
            abs(float(j) - float(h)) / span for j, h in zip(judge_points, human_points, strict=True)
        )
    if metric is Metric.ACCURACY:
        return accuracy([_label(v) for v in judge_points], [_label(v) for v in human_points])
    if metric is Metric.COHEN_KAPPA:
        return cohen_kappa([_label(v) for v in judge_points], [_label(v) for v in human_points])
    if metric is Metric.KRIPPENDORFF_ALPHA:
        level = "interval" if alignment.numeric else "nominal"
        units = [[j, h] for j, h in zip(judge_points, human_points, strict=True)]
        return krippendorff_alpha(units, level=level)
    if metric in (Metric.SPEARMAN, Metric.KENDALL, Metric.ICC):
        if not alignment.numeric:
            return None
        judge_numbers = [float(v) for v in judge_points]
        human_numbers = [float(v) for v in human_points]
        if metric is Metric.SPEARMAN:
            return spearman(judge_numbers, human_numbers)
        if metric is Metric.KENDALL:
            return kendall_tau(judge_numbers, human_numbers)
        return icc([[j, h] for j, h in zip(judge_numbers, human_numbers, strict=True)])
    if metric in (Metric.EARTH_MOVERS_DISTANCE, Metric.JENSEN_SHANNON):
        support = alignment.support
        if not support:
            return None
        distances: list[float] = []
        for item_id in ids:
            judge_values = _normalise(alignment.judge[item_id], alignment.numeric)
            human_values = _normalise(alignment.human[item_id], alignment.numeric)
            item_support = support
            if alignment.numeric:
                # Averaging repeated scores may produce fractions outside the
                # original discrete support. Keep those values as point masses,
                # while retaining the scale endpoints for EMD normalization.
                item_support = sorted({float(v) for v in [*support, *judge_values, *human_values]})
            p = distribution(judge_values, item_support)
            q = distribution(human_values, item_support)
            if sum(p) == 0 or sum(q) == 0:
                continue
            value = (
                earth_movers_distance(p, q, item_support)
                if metric is Metric.EARTH_MOVERS_DISTANCE
                else jensen_shannon(p, q)
            )
            if value is not None:
                distances.append(value)
        return mean(distances) if distances else None
    return None


def _collapse(values: list, numeric: bool, binary: bool = False) -> float | str:
    """Reduce a distribution to the single point the classic metrics need."""
    if numeric:
        return mean(float(v) for v in values)
    if binary:
        return "1" if sum(float(v) for v in values) * 2 >= len(values) else "0"
    return Counter(_label(v) for v in values).most_common(1)[0][0]


def _normalise(values: list, numeric: bool) -> list:
    """Snap values onto the metric support so histograms line up."""
    if numeric:
        return [float(v) for v in values]
    return [_label(v) for v in values]


def _label(value) -> str:
    if isinstance(value, float):
        rounded = round(value)
        return str(int(rounded)) if abs(value - rounded) < 1e-9 else f"{value:.4g}"
    return str(value)


def build_human_baseline(
    config: RunConfig,
    item_ids: list[str],
    annotations: dict[str, HumanAnnotations],
) -> HumanBaselineReport:
    """Leave one annotator out, on the run's items only, with no size cutoff.

    Each metric is averaged over annotators for whom it is defined, weighted by
    their number of comparable items. Stratum metrics use that same *total*
    per-annotator weight, not their stratum sample count. Strata are fixed using
    all labels before any annotator is held out. Undefined metrics are excluded
    individually and their effective annotator counts are reported.
    """
    report = HumanBaselineReport()
    if config.meta_evaluation is None:
        report.notes.append("Human baseline requires meta_evaluation.")
        return report
    selected = {i: annotations[i] for i in dict.fromkeys(item_ids) if i in annotations}
    report.n_items = len(selected)
    if not selected:
        report.notes.append("No human labels match the evaluated items.")
        return report
    if any(row.annotators is None for row in selected.values()):
        report.notes.append(
            "Human baseline unavailable: every matched item needs explicit annotator IDs."
        )
        return report
    labels = {i: row.labels for i, row in selected.items()}
    templates = build_alignment(config, [], labels)
    metrics = config.meta_evaluation.metrics
    threshold = config.meta_evaluation.agreement_threshold
    stratify = config.meta_evaluation.stratify_by_agreement

    for template in templates:
        groups: dict[str, str] = {}
        for item_id in selected:
            agreement = human_agreement(template, item_id)
            if agreement is not None:
                groups[item_id] = HIGH if agreement >= threshold else LOW
        by_annotator: dict[str, Alignment] = {}
        for item_id, row in selected.items():
            if len(row.labels) < 2:
                continue
            assert row.annotators is not None
            if len(row.annotators) != len(row.labels) or len(set(row.annotators)) != len(
                row.annotators
            ):
                raise ValueError(f"{item_id}: labels need one distinct annotator ID each")
            values = template.human[item_id]
            for index, annotator in enumerate(row.annotators):
                if annotator not in by_annotator:
                    by_annotator[annotator] = template.model_copy(
                        update={"judge": {}, "human": {}, "points": {}}
                    )
                alignment = by_annotator[annotator]
                alignment.judge[item_id] = [values[index]]
                alignment.human[item_id] = values[:index] + values[index + 1 :]

        target = HumanTargetReport(target=template.target, n_annotators=len(by_annotator))
        for annotator, alignment in by_annotator.items():
            ids = alignment.items()
            individual = TargetReport(
                target=template.target, n_items=len(ids), metrics=_compute(alignment, ids, metrics)
            )
            if stratify:
                for stratum in (HIGH, LOW):
                    group_ids = [i for i in ids if groups.get(i) == stratum]
                    individual.stratum_sizes[stratum] = len(group_ids)
                    individual.strata[stratum] = (
                        _compute(alignment, group_ids, metrics)
                        if group_ids
                        else {metric.value: None for metric in metrics}
                    )
            target.per_annotator[annotator] = individual

        def weighted(
            metric: str, stratum: str | None = None, target: HumanTargetReport = target
        ) -> tuple[float | None, int]:
            pairs = []
            for individual in target.per_annotator.values():
                value = (
                    individual.metrics.get(metric)
                    if stratum is None
                    else individual.strata.get(stratum, {}).get(metric)
                )
                if value is not None:
                    pairs.append((individual.n_items, value))
            weight = sum(n for n, _ in pairs)
            value = sum(n * value for n, value in pairs) / weight if weight else None
            return value, len(pairs)

        for metric in metrics:
            target.metrics[metric.value], target.metric_annotators[metric.value] = weighted(
                metric.value
            )
            if target.metrics[metric.value] is None:
                target.unavailable[metric.value] = "no_defined_annotators"
        if stratify:
            for stratum in (HIGH, LOW):
                target.strata[stratum] = {}
                target.stratum_annotators[stratum] = {}
                target.stratum_unavailable[stratum] = {}
                for metric in metrics:
                    value, count = weighted(metric.value, stratum)
                    target.strata[stratum][metric.value] = value
                    target.stratum_annotators[stratum][metric.value] = count
                    if value is None:
                        target.stratum_unavailable[stratum][metric.value] = "no_defined_annotators"
        report.targets.append(target)
    report.available = any(target.n_annotators > 0 for target in report.targets)
    if not report.available:
        report.notes.append("Human baseline requires at least two annotator labels per item.")
    return report
