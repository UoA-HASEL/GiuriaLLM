"""Per-member and panel evaluation views, without additional model calls.

Votes are first combined within a presentation order, then across orders.
Ranking votes select a candidate; they do not manufacture a full ranking.
"""

from collections import Counter, defaultdict

from ..spec import Aggregation, ResultKind, RunConfig
from ..types import ItemResult, Judgement, Verdict
from . import aggregate_item


def vote(values: list, preferred=None):
    """Plurality; preserve first-voter order, or a specified tied choice."""
    counts = Counter(values)
    winners = [v for v, n in counts.items() if n == max(counts.values())]
    return preferred if preferred in winners else winners[0]


def choice(verdict: Verdict) -> int | None:
    if verdict.best_choice is not None:
        return verdict.best_choice
    return verdict.ranking[0] if verdict.ranking else None


def evaluation_views(config: RunConfig, result: ItemResult) -> dict[str, ItemResult]:
    """Build one view per member and, if configured, a complete-member panel.

    Raw judgements remain attached to their view for usage accounting. Samples
    are combined per member/order before members vote. A panel order requires
    a valid verdict from every configured member.
    """
    models = config.judges()
    by_member: dict[str, dict[int, Judgement]] = {}
    views: dict[str, ItemResult] = {}
    for model in models:
        raw = [j for j in result.judgements if j.model == model.display]
        grouped: dict[int, list[Judgement]] = defaultdict(list)
        for j in raw:
            if j.ok:
                grouped[j.variant.index].append(j)
        orders = {}
        for index, js in sorted(grouped.items()):
            js.sort(key=lambda j: j.sample)
            verdict = _combine(config, js, member=True)
            orders[index] = js[0].model_copy(update={"verdict": verdict})
        by_member[model.display] = orders
        views[model.display] = _finish(config, result.item_id, raw, orders)

    if config.panel:
        indices = sorted({i for orders in by_member.values() for i in orders})
        orders = {}
        for index in indices:
            js = [by_member[m.display][index] for m in models if index in by_member[m.display]]
            if len(js) != len(models):
                continue
            verdict = _combine(config, js)
            orders[index] = js[0].model_copy(update={"model": "panel", "verdict": verdict})
        views["panel"] = _finish(config, result.item_id, result.judgements, orders)
    return views


def _combine(config: RunConfig, js: list[Judgement], member: bool = False) -> Verdict:
    kind = config.judge.outputs.result
    if (
        kind is ResultKind.RANKING
        and config.panel
        and config.panel.aggregation is Aggregation.MAJORITY_VOTE
    ):
        # This is an internal selection, not a replacement for the raw ranking.
        return Verdict(best_choice=vote([choice(j.verdict) for j in js]))
    cfg = config
    if member and config.panel:
        # Do not apply panel member weights to a member's own samples.
        cfg = config.model_copy(update={"panel": config.panel.model_copy(update={"weights": {}})})
    verdict, _ = aggregate_item(cfg, js)
    return verdict


def _finish(
    config: RunConfig, item_id: str, raw: list[Judgement], orders: dict[int, Judgement]
) -> ItemResult:
    out = ItemResult(item_id=item_id, judgements=raw)
    if not orders:
        return out
    if config.judge.outputs.result is ResultKind.GRADED:
        out.aggregate, out.disagreement = aggregate_item(config, list(orders.values()))
        return out
    out.order_choices = {i: choice(j.verdict) for i, j in orders.items()}
    out.selected_candidate = vote(list(out.order_choices.values()), out.order_choices.get(0))
    out.flipped = len(set(out.order_choices.values())) > 1 if len(orders) >= 2 else None
    if config.judge.outputs.result is ResultKind.BEST_CHOICE:
        representative = next(
            j.verdict for j in orders.values() if choice(j.verdict) == out.selected_candidate
        )
        out.aggregate = representative.model_copy(update={"best_choice": out.selected_candidate})
    elif config.panel is None or config.panel.aggregation is not Aggregation.MAJORITY_VOTE:
        # Preserve the existing full-ranking aggregation for non-voting runs.
        out.aggregate, out.disagreement = aggregate_item(config, raw)
        out.selected_candidate = choice(out.aggregate) if out.aggregate else None
    return out


def distribution_view(view: ItemResult, members: list[ItemResult]) -> ItemResult:
    """For distribution metrics, one final value per member, not per API call."""
    if view.aggregate is None and view.selected_candidate is None:
        # Partial member answers cannot manufacture a missing panel verdict.
        return view.model_copy(update={"judgements": []})
    js = []
    for member in members:
        verdict = member.aggregate
        if member.selected_candidate is not None:
            verdict = Verdict(best_choice=member.selected_candidate)
        if verdict is not None:
            source = next((j for j in member.judgements if j.ok), None)
            if source:
                js.append(source.model_copy(update={"verdict": verdict}))
    return view.model_copy(update={"judgements": js})
