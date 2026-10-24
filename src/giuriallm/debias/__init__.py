"""Bias mitigation as infrastructure.

Position swapping (pairwise) and permutation (listwise) are produced by the
framework, default-on, and the resulting *flip rate* -- how often a verdict
changes when only the presentation order changes -- is reported as a first-class
diagnostic rather than left to each study to rediscover.
"""

from __future__ import annotations

from collections import Counter
from itertools import permutations

from ..spec import DebiasSpec, EvaluationType, RunConfig
from ..types import ItemResult, Judgement, Variant


def variants_for(config: RunConfig, n_candidates: int) -> list[Variant]:
    """The presentation orders an item will be judged under."""
    debias: DebiasSpec = config.debias
    identity = list(range(n_candidates))

    if config.judge.type is EvaluationType.POINTWISE:
        return [Variant(order=identity, index=0)]

    if config.judge.type is EvaluationType.PAIRWISE:
        swap = debias.position_swap is not False  # None means automatic: on
        orders = [identity, [1, 0]] if swap else [identity]
        return [Variant(order=o, index=i) for i, o in enumerate(orders)]

    return [
        Variant(order=list(o), index=i)
        for i, o in enumerate(_listwise_orders(n_candidates, debias.permutations))
    ]


def _listwise_orders(n: int, count: int) -> list[tuple[int, ...]]:
    """Deterministic, well-spread permutations: identity, reverse, then rotations."""
    identity = tuple(range(n))
    chosen: list[tuple[int, ...]] = [identity]
    if count > 1:
        chosen.append(tuple(reversed(identity)))
    shift = 1
    while len(chosen) < count:
        rotated = identity[shift:] + identity[:shift]
        if rotated not in chosen:
            chosen.append(rotated)
        shift += 1
        if shift >= n:  # exhausted rotations; fall back to lexicographic order
            for candidate in permutations(identity):
                if candidate not in chosen:
                    chosen.append(candidate)
                if len(chosen) >= count:
                    break
            break
    return chosen[:count]


def verdict_key(judgement: Judgement) -> tuple | None:
    """A comparable summary of a verdict, used to detect order-induced flips."""
    verdict = judgement.verdict
    if verdict is None:
        return None
    if verdict.best_choice is not None:
        return ("best_choice", verdict.best_choice)
    if verdict.ranking is not None:
        return ("ranking", tuple(verdict.ranking))
    return ("scores", tuple((s.criterion, s.score) for s in verdict.scores))


def flipped(judgements: list[Judgement]) -> bool | None:
    """Did the verdict change across presentation orders, holding model+sample fixed?

    Returns ``None`` when there is nothing to compare (a single order, or too few
    successful judgements).
    """
    groups: dict[tuple[str, int], set[tuple]] = {}
    orders: set[tuple[int, ...]] = set()
    for j in judgements:
        if not j.ok:
            continue
        key = verdict_key(j)
        if key is None:
            continue
        orders.add(tuple(j.variant.order))
        groups.setdefault((j.model, j.sample), set()).add(key)

    if len(orders) < 2:
        return None
    comparable = [values for values in groups.values() if len(values) >= 1]
    if not comparable:
        return None
    return any(len(values) > 1 for values in comparable)


def flip_rate(results: list[ItemResult]) -> float | None:
    """Fraction of items whose verdict changed under a different candidate order."""
    flags = [r.flipped for r in results if r.flipped is not None]
    if not flags:
        return None
    return sum(1 for f in flags if f) / len(flags)


def position_bias(results: list[ItemResult]) -> dict[str, float] | None:
    """How often each *shown position* won, over all pairwise/listwise judgements.

    A judge with no positional preference wins each position equally often; a
    skew here is the bias that position swapping is meant to average out.
    """
    counts: Counter[int] = Counter()
    total = 0
    for result in results:
        for j in result.judgements:
            if not j.ok or j.verdict is None:
                continue
            winner: int | None = None
            if j.verdict.best_choice is not None:
                winner = j.verdict.best_choice
            elif j.verdict.ranking:
                winner = j.verdict.ranking[0]
            if winner is None:
                continue
            counts[j.variant.order.index(winner)] += 1
            total += 1
    if total == 0:
        return None
    return {f"position_{pos + 1}": counts[pos] / total for pos in sorted(counts)}


def self_preference(results: list[ItemResult], candidate_models: list[str]) -> dict[str, float]:
    """Win rate of each candidate slot, by judging model.

    When candidate slots correspond to generating models, a judge that favours
    its own output shows up as an elevated win rate for its own slot.
    """
    wins: Counter[tuple[str, int]] = Counter()
    totals: Counter[str] = Counter()
    for result in results:
        for j in result.judgements:
            if not j.ok or j.verdict is None or j.verdict.best_choice is None:
                continue
            wins[(j.model, j.verdict.best_choice)] += 1
            totals[j.model] += 1

    rates: dict[str, float] = {}
    for (judge_model, slot), count in wins.items():
        slot_name = candidate_models[slot] if slot < len(candidate_models) else f"slot_{slot}"
        rates[f"{judge_model} -> {slot_name}"] = count / max(1, totals[judge_model])
    return rates
