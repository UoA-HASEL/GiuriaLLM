"""Execution engine.

Fans judgement tasks out across items x judges x samples x debias variants,
runs them with bounded concurrency and retries, serves repeats from a
content-addressed cache, and records what every call cost.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from ..aggregate import aggregate_item, bradley_terry, pairwise_comparisons
from ..aggregate.evaluation import distribution_view, evaluation_views
from ..dataset import load_human_annotations, load_items
from ..debias import flip_rate, flipped, position_bias, variants_for
from ..evidence import EvidenceProvider, build_evidence
from ..metaeval import AgreementReport, build_alignment, build_human_baseline, evaluate_alignment
from ..prompts import TOOL_NAME, PromptCompiler, VerdictParseError, parse_verdict
from ..providers import LLMRequest, Provider, ProviderError, get_provider, price_note
from ..spec import EvaluationType, ModelSpec, RunConfig, config_hash
from ..types import EvaluationItem, ItemResult, Judgement, Status, Usage, Variant
from .cache import ResponseCache
from .cost import account, co2_grams


class EvaluationSummary(BaseModel):
    n_items: int
    n_valid: int
    n_failed: int
    agreement: AgreementReport | None = None
    usage: Usage = Field(default_factory=Usage)
    total_tokens: int = 0
    latency_per_item_mean_s: float | None = None
    flip_rate: float | None = None
    n_flip_items: int = 0
    n_flipped: int = 0


class RunResult(BaseModel):
    run_id: str
    config_hash: str
    started_at: str
    finished_at: str | None = None
    results: list[ItemResult] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    flip_rate: float | None = None
    position_bias: dict[str, float] | None = None
    bradley_terry: dict[str, float] | None = None
    agreement: AgreementReport | None = None
    notes: list[str] = Field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0
    evaluations: dict[str, EvaluationSummary] = Field(default_factory=dict)
    human_baseline: dict | None = None

    @property
    def n_items(self) -> int:
        return len(self.results)

    @property
    def n_judgements(self) -> int:
        return sum(len(r.judgements) for r in self.results)

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.results for j in r.judgements if not j.ok)

    @property
    def co2_g(self) -> float:
        return co2_grams(self.usage.energy_wh)


class Engine:
    def __init__(self, config: RunConfig, provider: Provider | None = None):
        self.config = config
        self.compiler = PromptCompiler(config.judge, config.prompt)
        self.cache = ResponseCache(config.engine.cache_dir, enabled=config.engine.cache)
        self._providers: dict[str, Provider] = {}
        if provider is not None:
            self._providers[provider.name] = provider
        self.evidence: list[EvidenceProvider] = build_evidence(config.evidence)
        self.notes: list[str] = []

    def provider_for(self, model: ModelSpec) -> Provider:
        if model.provider not in self._providers:
            self._providers[model.provider] = get_provider(model.provider)
        return self._providers[model.provider]

    # -- public API ------------------------------------------------------

    async def run(self, items: list[EvaluationItem] | None = None) -> RunResult:
        started = datetime.now(UTC)
        if items is None:
            items = load_items(self.config.dataset, self.config.judge.reference.mode)

        result = RunResult(
            run_id=f"{started:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}",
            config_hash=config_hash(self.config),
            started_at=started.isoformat(),
        )

        semaphore = asyncio.Semaphore(self.config.engine.concurrency)
        item_results = await asyncio.gather(*(self._judge_item(item, semaphore) for item in items))
        views: dict[str, list[ItemResult]] = {m.display: [] for m in self.config.judges()}
        if self.config.panel:
            views["panel"] = []
        for item_result in item_results:
            for name, view in evaluation_views(self.config, item_result).items():
                views[name].append(view)
        primary = "panel" if self.config.panel else self.config.model.display
        result.results = views[primary]

        total = Usage()
        for item_result in result.results:
            total = total + item_result.usage
        result.usage = total

        result.flip_rate = flip_rate(result.results)
        result.position_bias = position_bias(result.results)

        if self.config.judge.type is EvaluationType.PAIRWISE:
            slots = self.config.dataset.fields.candidates
            comparisons = pairwise_comparisons(result.results, slots)
            if comparisons:
                result.bradley_terry = bradley_terry(comparisons)

        annotations = (
            load_human_annotations(self.config.meta_evaluation.human_labels)
            if self.config.meta_evaluation
            else None
        )
        for name, view_items in views.items():
            usage = Usage()
            for view in view_items:
                usage = usage + view.usage
            comparable = [v for v in view_items if v.flipped is not None]
            summary = EvaluationSummary(
                n_items=len(view_items),
                n_valid=sum(
                    v.aggregate is not None or v.selected_candidate is not None for v in view_items
                ),
                n_failed=sum(not j.ok for v in view_items for j in v.judgements),
                usage=usage,
                total_tokens=usage.total_tokens,
                latency_per_item_mean_s=(usage.latency_s / len(view_items) if view_items else None),
                flip_rate=flip_rate(view_items) if self.config.debias.report_flip_rate else None,
                n_flip_items=len(comparable),
                n_flipped=sum(v.flipped for v in comparable),
            )
            if annotations is not None:
                metric_views = []
                for i, view in enumerate(view_items):
                    members = (
                        [views[m.display][i] for m in self.config.judges()]
                        if name == "panel"
                        else [view]
                    )
                    metric_views.append(distribution_view(view, members))
                labels = {i: a.labels for i, a in annotations.items()}
                alignments = build_alignment(self.config, metric_views, labels)
                meta = self.config.meta_evaluation
                summary.agreement = evaluate_alignment(
                    alignments,
                    meta.metrics,
                    stratify=meta.stratify_by_agreement,
                    threshold=meta.agreement_threshold,
                )
            result.evaluations[name] = summary
        result.agreement = result.evaluations[primary].agreement
        if annotations is not None:
            result.human_baseline = build_human_baseline(
                self.config,
                [i.id for i in items],
                annotations,
            ).model_dump(mode="json")

        result.cache_hits = self.cache.hits
        result.cache_misses = self.cache.misses
        result.notes = self._collect_notes(result)
        result.finished_at = datetime.now(UTC).isoformat()
        return result

    async def aclose(self) -> None:
        self.cache.close()
        for provider in self._providers.values():
            await provider.aclose()

    # -- internals -------------------------------------------------------

    def _collect_notes(self, result: RunResult) -> list[str]:
        notes = list(self.notes)
        for provider in self._providers.values():
            notes.extend(getattr(provider, "notes", []))
        for model in self.config.judges():
            note = price_note(model)
            if note is not None and note not in notes:
                notes.append(note)
        if result.n_failed:
            notes.append(
                f"{result.n_failed} of {result.n_judgements} judgements failed and are "
                "excluded from aggregates; see the manifest for each failure"
            )
        if any(p.name == "mock" for p in self._providers.values()):
            notes.append(
                "provider=mock produced these verdicts offline; they are for pipeline "
                "testing only and carry no evaluative meaning"
            )
        return notes

    async def _judge_item(self, item: EvaluationItem, semaphore: asyncio.Semaphore) -> ItemResult:
        evidence = await self._gather_evidence(item)
        variants = variants_for(self.config, len(item.candidates))

        tasks = [
            self._judge_once(item, model, sample, variant, evidence, semaphore)
            for model in self.config.judges()
            for sample in range(model.samples)
            for variant in variants
        ]
        judgements = list(await asyncio.gather(*tasks))

        aggregate, disagreement = aggregate_item(self.config, judgements)
        return ItemResult(
            item_id=item.id,
            aggregate=aggregate,
            judgements=judgements,
            flipped=flipped(judgements),
            disagreement=disagreement,
        )

    async def _gather_evidence(self, item: EvaluationItem) -> dict[str, object]:
        evidence: dict[str, object] = {}
        for provider in self.evidence:
            evidence[provider.name] = await provider.gather(item)
        return evidence

    async def _judge_once(
        self,
        item: EvaluationItem,
        model: ModelSpec,
        sample: int,
        variant: Variant,
        evidence: dict[str, object],
        semaphore: asyncio.Semaphore,
    ) -> Judgement:
        prompt = self.compiler.compile(item, variant, evidence)
        request = LLMRequest(
            model=model,
            system=prompt.system,
            user=prompt.user,
            schema=prompt.schema,
            tool_name=TOOL_NAME,
            sample=sample,
        )
        judgement = Judgement(
            item_id=item.id,
            model=model.display,
            sample=sample,
            variant=variant,
            evidence=dict(evidence),
        )

        cached = self.cache.get(request.cache_key)
        if cached is not None:
            response, latency, from_cache = cached, 0.0, True
        else:
            async with semaphore:
                try:
                    response, latency = await self._call_with_retries(model, request)
                except ProviderError as exc:
                    judgement.status = Status.PROVIDER_ERROR
                    judgement.error = str(exc)
                    return judgement
            from_cache = False
            self.cache.put(request.cache_key, response)

        judgement.usage = account(
            response.usage,
            model,
            self.config.engine.energy_wh_per_ktok,
            latency_s=latency,
            cached=from_cache,
        )
        judgement.raw_output = response.text or None

        if response.payload is None:
            judgement.status = Status.PARSE_ERROR
            judgement.error = (
                f"the model returned no structured verdict (stop_reason={response.stop_reason!r})"
            )
            return judgement

        try:
            judgement.verdict = parse_verdict(
                response.payload, self.config.judge, variant, prompt.labels
            )
        except VerdictParseError as exc:
            judgement.status = Status.PARSE_ERROR
            judgement.error = str(exc)
        return judgement

    async def _call_with_retries(self, model: ModelSpec, request: LLMRequest):
        provider = self.provider_for(model)
        attempts = self.config.engine.max_retries + 1
        last: ProviderError | None = None
        for attempt in range(attempts):
            start = time.monotonic()
            try:
                response = await provider.complete(request)
                return response, time.monotonic() - start
            except ProviderError as exc:
                last = exc
                if not exc.retryable or attempt == attempts - 1:
                    raise
                await asyncio.sleep(min(8.0, 0.5 * 2**attempt))
        raise last or ProviderError("request failed with no error recorded")


async def run_config(config: RunConfig, provider: Provider | None = None) -> RunResult:
    """Convenience wrapper: build an engine, run it, close it."""
    engine = Engine(config, provider=provider)
    try:
        return await engine.run()
    finally:
        await engine.aclose()
