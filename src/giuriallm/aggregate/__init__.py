"""Aggregation: from many judgements to one verdict.

Covers self-consistency (n samples of one model), panels of judges (several
models), and the run-level estimators -- Bradley-Terry for turning pairwise
outcomes into a ranking, Dawid-Skene for recovering latent labels and per-judge
reliability.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import mean, median, pstdev

from ..spec import Aggregation, Criterion, PanelSpec, ResultKind, RunConfig, ScaleKind
from ..types import CriterionScore, Judgement, Verdict


def aggregate_item(
    config: RunConfig, judgements: list[Judgement]
) -> tuple[Verdict | None, float | None]:
    """Combine the successful judgements for one item into a single verdict."""
    usable = [j for j in judgements if j.ok and j.verdict is not None]
    if not usable:
        return None, None

    method = config.panel.aggregation if config.panel else Aggregation.MEAN
    weights = _weights(config.panel, usable)

    if config.judge.outputs.result is ResultKind.GRADED:
        return _aggregate_graded(config, usable, method, weights)
    if config.judge.outputs.result is ResultKind.BEST_CHOICE:
        return _aggregate_choice(usable, weights)
    return _aggregate_ranking(usable, weights)


def _first_explanation(judgements: list[Judgement]) -> str | None:
    """Carry one representative explanation onto the aggregate.

    Every individual explanation stays in the manifest; the aggregate keeps one
    so a reader is not left with a bare number.
    """
    return next(
        (j.verdict.explanation for j in judgements if j.verdict and j.verdict.explanation), None
    )


def _weights(panel: PanelSpec | None, judgements: list[Judgement]) -> dict[str, float]:
    if panel is None or not panel.weights:
        return {j.model: 1.0 for j in judgements}
    return {j.model: panel.weights.get(j.model, 1.0) for j in judgements}


def _aggregate_graded(
    config: RunConfig,
    judgements: list[Judgement],
    method: Aggregation,
    weights: dict[str, float],
) -> tuple[Verdict, float | None]:
    scores: list[CriterionScore] = []
    spreads: list[float] = []

    for criterion in config.judge.criteria:
        values = [
            (j.verdict.score_for(criterion.name), weights.get(j.model, 1.0))
            for j in judgements
            if j.verdict is not None and j.verdict.score_for(criterion.name) is not None
        ]
        if not values:
            continue

        if criterion.scale.kind is ScaleKind.CATEGORICAL or method is Aggregation.MAJORITY_VOTE:
            tally: Counter = Counter()
            for value, weight in values:
                tally[value] += weight
            winner, top = tally.most_common(1)[0]
            scores.append(CriterionScore(criterion=criterion.name, score=winner))
            spreads.append(1.0 - top / sum(tally.values()))
        else:
            numbers = [float(v) for v, _ in values]
            ws = [w for _, w in values]
            if method is Aggregation.MEDIAN:
                point = median(numbers)
            elif method is Aggregation.WEIGHTED and sum(ws) > 0:
                point = sum(n * w for n, w in zip(numbers, ws, strict=True)) / sum(ws)
            else:
                point = mean(numbers)
            scores.append(CriterionScore(criterion=criterion.name, score=point))
            spreads.append(_normalised_spread(numbers, criterion))

    explanation = _first_explanation(judgements)
    feedback = next(
        (j.verdict.feedback for j in judgements if j.verdict and j.verdict.feedback), None
    )
    verdict = Verdict(scores=scores, explanation=explanation, feedback=feedback)
    return verdict, (mean(spreads) if spreads else None)


def _normalised_spread(values: list[float], criterion: Criterion) -> float:
    if len(values) < 2:
        return 0.0
    spread = pstdev(values)
    scale = criterion.scale
    if scale.min is not None and scale.max is not None and scale.max > scale.min:
        return spread / (scale.max - scale.min)
    return spread


def _aggregate_choice(
    judgements: list[Judgement], weights: dict[str, float]
) -> tuple[Verdict, float | None]:
    tally: Counter[int] = Counter()
    for j in judgements:
        if j.verdict is not None and j.verdict.best_choice is not None:
            tally[j.verdict.best_choice] += weights.get(j.model, 1.0)
    if not tally:
        return Verdict(), None
    winner, top = tally.most_common(1)[0]
    total = sum(tally.values())
    explanation = _first_explanation(judgements)
    return Verdict(best_choice=winner, explanation=explanation), 1.0 - top / total


def _aggregate_ranking(
    judgements: list[Judgement], weights: dict[str, float]
) -> tuple[Verdict, float | None]:
    """Borda count: average weighted rank position, lowest first."""
    positions: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for j in judgements:
        if j.verdict is None or not j.verdict.ranking:
            continue
        weight = weights.get(j.model, 1.0)
        for position, candidate in enumerate(j.verdict.ranking):
            positions[candidate].append((float(position), weight))
    if not positions:
        return Verdict(), None

    averaged = {
        candidate: sum(p * w for p, w in entries) / sum(w for _, w in entries)
        for candidate, entries in positions.items()
    }
    ranking = sorted(averaged, key=lambda c: (averaged[c], c))
    tops = Counter(j.verdict.ranking[0] for j in judgements if j.verdict and j.verdict.ranking)
    disagreement = 1.0 - tops.most_common(1)[0][1] / sum(tops.values()) if tops else None
    explanation = _first_explanation(judgements)
    return Verdict(ranking=ranking, explanation=explanation), disagreement


# --------------------------------------------------------------------------
# Run-level estimators
# --------------------------------------------------------------------------


def bradley_terry(
    comparisons: list[tuple[str, str]],
    iterations: int = 200,
    tolerance: float = 1e-8,
) -> dict[str, float]:
    """Bradley-Terry strengths from ``(winner, loser)`` pairs.

    Fitted by minorisation-maximisation. Strengths are normalised to sum to the
    number of players, so 1.0 is average.
    """
    players = sorted({p for pair in comparisons for p in pair})
    if not players:
        return {}

    wins: Counter[str] = Counter()
    matches: dict[tuple[str, str], int] = defaultdict(int)
    for winner, loser in comparisons:
        wins[winner] += 1
        key = (winner, loser) if winner < loser else (loser, winner)
        matches[key] += 1

    strength = {p: 1.0 for p in players}
    for _ in range(iterations):
        updated: dict[str, float] = {}
        for p in players:
            denominator = 0.0
            for (a, b), count in matches.items():
                if p not in (a, b):
                    continue
                other = b if p == a else a
                denominator += count / (strength[p] + strength[other])
            updated[p] = wins[p] / denominator if denominator > 0 else strength[p]

        total = sum(updated.values())
        if total <= 0:
            break
        updated = {p: v * len(players) / total for p, v in updated.items()}
        delta = max(abs(updated[p] - strength[p]) for p in players)
        strength = updated
        if delta < tolerance:
            break
    return strength


def dawid_skene(
    labels: dict[str, dict[str, str]],
    iterations: int = 100,
    tolerance: float = 1e-6,
) -> tuple[dict[str, str], dict[str, float]]:
    """Dawid-Skene EM over ``{item: {annotator: label}}``.

    Returns the inferred label per item and each annotator's estimated accuracy.
    Useful for a panel of judges: it weights members by measured reliability
    instead of counting every member's vote equally.
    """
    items = sorted(labels)
    annotators = sorted({a for row in labels.values() for a in row})
    classes = sorted({label for row in labels.values() for label in row.values()})
    if not items or not classes:
        return {}, {}

    n_classes = len(classes)
    posterior: dict[str, dict[str, float]] = {}
    for item in items:
        tally = Counter(labels[item].values())
        total = sum(tally.values())
        posterior[item] = {c: tally.get(c, 0) / total for c in classes}

    prior = {c: 1.0 / n_classes for c in classes}
    confusion: dict[str, dict[str, dict[str, float]]] = {}

    for _ in range(iterations):
        # M-step: class priors and per-annotator confusion matrices.
        prior = {c: sum(posterior[i][c] for i in items) / len(items) for c in classes}
        confusion = {}
        for annotator in annotators:
            table = {c: {k: 1e-9 for k in classes} for c in classes}
            for item in items:
                observed = labels[item].get(annotator)
                if observed is None:
                    continue
                for true_class in classes:
                    table[true_class][observed] += posterior[item][true_class]
            confusion[annotator] = {
                true_class: {k: v / sum(row.values()) for k, v in row.items()}
                for true_class, row in table.items()
            }

        # E-step: posterior over true classes.
        delta = 0.0
        for item in items:
            unnormalised: dict[str, float] = {}
            for true_class in classes:
                likelihood = math.log(max(prior[true_class], 1e-12))
                for annotator, observed in labels[item].items():
                    likelihood += math.log(max(confusion[annotator][true_class][observed], 1e-12))
                unnormalised[true_class] = likelihood
            peak = max(unnormalised.values())
            weights = {c: math.exp(v - peak) for c, v in unnormalised.items()}
            total = sum(weights.values())
            new = {c: w / total for c, w in weights.items()}
            delta = max(delta, max(abs(new[c] - posterior[item][c]) for c in classes))
            posterior[item] = new
        if delta < tolerance:
            break

    inferred = {item: max(posterior[item], key=lambda c: posterior[item][c]) for item in items}
    accuracy = {
        annotator: sum(confusion[annotator][c][c] * prior[c] for c in classes)
        for annotator in annotators
    }
    return inferred, accuracy


def pairwise_comparisons(results, slot_names: list[str]) -> list[tuple[str, str]]:
    """Extract ``(winner, loser)`` pairs from pairwise item results."""
    pairs: list[tuple[str, str]] = []
    for result in results:
        verdict = result.aggregate
        if verdict is None or verdict.best_choice is None:
            continue
        winner = verdict.best_choice
        for slot in range(len(slot_names)):
            if slot != winner:
                pairs.append((slot_names[winner], slot_names[slot]))
    return pairs
