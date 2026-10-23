"""End-to-end engine behaviour, using scripted and mock providers."""

from __future__ import annotations

import json

import pytest

from giuriallm.engine import Engine, run_config, write_run
from giuriallm.engine.cost import account
from giuriallm.engine.manifest import read_manifest
from giuriallm.providers import (
    LLMRequest,
    LLMResponse,
    MockProvider,
    Provider,
    ProviderError,
    RawUsage,
)
from giuriallm.report import write_reports
from giuriallm.spec import ModelSpec, parse_config
from giuriallm.types import Status


def write_dataset(tmp_path, rows, name="data.jsonl"):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def pairwise_config(tmp_path, **overrides):
    dataset = write_dataset(
        tmp_path,
        [
            {"id": "i1", "prompt": "task one", "gen_a": "a1", "gen_b": "b1"},
            {"id": "i2", "prompt": "task two", "gen_a": "a2", "gen_b": "b2"},
        ],
    )
    payload = {
        "judge": {
            "type": "pairwise",
            "criteria": [{"name": "correctness", "definition": "works?"}],
            "outputs": {"result": "best_choice"},
        },
        "model": {"provider": "mock", "id": "mock-1"},
        "dataset": {"path": str(dataset), "fields": {"candidates": ["gen_a", "gen_b"]}},
        "engine": {"cache": False, "cache_dir": str(tmp_path / "cache")},
        "report": {"output_dir": str(tmp_path / "runs")},
    }
    payload.update(overrides)
    return parse_config(payload)


class ScriptedProvider(Provider):
    """Returns queued payloads in order, recording every request it received."""

    name = "mock"

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        payload = self.payloads.pop(0) if self.payloads else None
        if isinstance(payload, Exception):
            raise payload
        return LLMResponse(
            payload=payload,
            text="scripted",
            usage=RawUsage(input_tokens=100, output_tokens=20),
            model=request.model.id,
        )


async def test_run_produces_one_judgement_per_variant_and_sample(tmp_path):
    config = pairwise_config(tmp_path, model={"provider": "mock", "id": "mock-1", "samples": 2})
    result = await run_config(config, provider=MockProvider())
    # 2 items x 2 samples x 2 position variants
    assert result.n_items == 2
    assert result.n_judgements == 8
    assert result.n_failed == 0


async def test_position_swap_flip_is_detected_and_reported(tmp_path):
    """A judge that always names the first-shown candidate flips on every item."""
    config = pairwise_config(tmp_path)
    provider = ScriptedProvider([{"explanation": "first looks better", "best_choice": "A"}] * 4)
    result = await run_config(config, provider=provider)
    assert result.flip_rate == pytest.approx(1.0)
    assert result.position_bias == {"position_1": 1.0}


async def test_stable_judge_has_zero_flip_rate(tmp_path):
    config = pairwise_config(tmp_path)
    # Under order [0,1] pick A; under order [1,0] pick B -- both mean candidate 0.
    provider = ScriptedProvider(
        [
            {"explanation": "x", "best_choice": "A"},
            {"explanation": "x", "best_choice": "B"},
        ]
        * 2
    )
    result = await run_config(config, provider=provider)
    assert result.flip_rate == pytest.approx(0.0)
    assert all(r.aggregate.best_choice == 0 for r in result.results)


async def test_parse_failure_is_recorded_not_coerced(tmp_path):
    config = pairwise_config(tmp_path)
    provider = ScriptedProvider(
        [
            {"explanation": "fine", "best_choice": "A"},
            {"explanation": "fine", "best_choice": "NOT_A_LABEL"},
            {"explanation": "fine", "best_choice": "B"},
            None,
        ]
    )
    result = await run_config(config, provider=provider)
    statuses = [j.status for r in result.results for j in r.judgements]
    assert statuses.count(Status.PARSE_ERROR) == 2
    failed = [j for r in result.results for j in r.judgements if not j.ok]
    assert all(j.verdict is None and j.error for j in failed)
    # The surviving judgements still produce an aggregate.
    assert any(r.aggregate is not None for r in result.results)


async def test_provider_error_is_captured_per_judgement(tmp_path):
    config = pairwise_config(tmp_path)
    provider = ScriptedProvider(
        [
            ProviderError("boom", retryable=False),
            {"explanation": "fine", "best_choice": "A"},
            {"explanation": "fine", "best_choice": "A"},
            {"explanation": "fine", "best_choice": "A"},
        ]
    )
    result = await run_config(config, provider=provider)
    errors = [j for r in result.results for j in r.judgements if j.status is Status.PROVIDER_ERROR]
    assert len(errors) == 1
    assert "boom" in errors[0].error


async def test_retry_then_succeed(tmp_path):
    config = pairwise_config(
        tmp_path, engine={"cache": False, "max_retries": 2, "cache_dir": str(tmp_path / "c")}
    )
    provider = ScriptedProvider(
        [ProviderError("overloaded", retryable=True)]
        + [{"explanation": "fine", "best_choice": "A"}] * 4
    )
    result = await run_config(config, provider=provider)
    assert result.n_failed == 0
    assert len(provider.requests) == 5  # one retried call


async def test_cache_serves_a_repeat_run_without_calling_the_provider(tmp_path):
    config = pairwise_config(tmp_path, engine={"cache": True, "cache_dir": str(tmp_path / "cache")})
    first = await run_config(config, provider=MockProvider())
    assert first.cache_misses == 4 and first.cache_hits == 0

    provider = ScriptedProvider([])  # would raise IndexError if it were called
    second = await run_config(config, provider=provider)
    assert second.cache_hits == 4
    assert provider.requests == []
    assert second.usage.cost_usd == 0.0
    assert [r.aggregate.best_choice for r in first.results] == [
        r.aggregate.best_choice for r in second.results
    ]


async def test_cache_key_changes_with_the_prompt(tmp_path):
    config = pairwise_config(tmp_path, engine={"cache": True, "cache_dir": str(tmp_path / "c")})
    await run_config(config, provider=MockProvider())

    changed = parse_config(
        {
            **json.loads(config.model_dump_json()),
            "prompt": {"strategy": "direct", "order": "explain_then_rate"},
        }
    )
    second = await run_config(changed, provider=MockProvider())
    assert second.cache_hits == 0


async def test_mock_provider_is_deterministic(tmp_path):
    config = pairwise_config(tmp_path)
    a = await run_config(config, provider=MockProvider())
    b = await run_config(config, provider=MockProvider())
    assert [r.aggregate.best_choice for r in a.results] == [
        r.aggregate.best_choice for r in b.results
    ]


async def test_manifest_and_reports_are_written(tmp_path):
    config = pairwise_config(tmp_path)
    result = await run_config(config, provider=MockProvider())
    run_dir = write_run(config, result, config.report.output_dir)
    write_reports(config, result, run_dir)

    manifest = read_manifest(run_dir)
    assert manifest["config_hash"] == result.config_hash
    assert manifest["counts"]["judgements"] == result.n_judgements
    assert manifest["usage"]["energy_is_estimated"] is True
    assert (run_dir / "config.yaml").is_file()

    judgements = (run_dir / "judgements.jsonl").read_text().strip().splitlines()
    assert len(judgements) == result.n_judgements
    assert json.loads(judgements[0])["variant"]["order"] in ([0, 1], [1, 0])

    report = (run_dir / "report.md").read_text()
    assert "Flip rate" in report
    assert "Estimated energy (Wh/item)" in report


async def test_evidence_reaches_the_prompt(tmp_path):
    config = pairwise_config(tmp_path, evidence=[{"name": "python_syntax"}])
    provider = ScriptedProvider([{"explanation": "x", "best_choice": "A"}] * 4)
    await run_config(config, provider=provider)
    assert "python_syntax" in provider.requests[0].user


async def test_meta_evaluation_runs_and_stratifies(tmp_path):
    labels = tmp_path / "human.jsonl"
    labels.write_text(
        json.dumps({"id": "i1", "labels": [0, 0, 0, 0]})
        + "\n"
        + json.dumps({"id": "i2", "labels": [0, 1, 0, 1]})
        + "\n"
    )
    config = pairwise_config(
        tmp_path,
        meta_evaluation={
            "human_labels": str(labels),
            "metrics": ["accuracy", "cohen_kappa", "jensen_shannon", "earth_movers_distance"],
            "stratify_by_agreement": True,
        },
    )
    result = await run_config(config, provider=MockProvider())
    agreement = result.agreement
    assert agreement is not None and agreement.n_matched == 2
    target = agreement.targets[0]
    assert set(target.metrics) == {
        "accuracy",
        "cohen_kappa",
        "jensen_shannon",
        "earth_movers_distance",
    }
    # i1 is unanimous, i2 is split down the middle.
    assert target.stratum_sizes == {"high_agreement": 1, "low_agreement": 1}


def test_cost_accounting_uses_list_prices():
    usage = account(
        RawUsage(input_tokens=1_000_000, output_tokens=1_000_000),
        ModelSpec(provider="anthropic", id="claude-opus-5"),
        energy_wh_per_ktok=0.4,
    )
    assert usage.cost_usd == pytest.approx(30.0)
    assert usage.energy_wh == pytest.approx(800.0)


def test_cached_calls_cost_nothing():
    usage = account(
        RawUsage(input_tokens=1000, output_tokens=1000),
        ModelSpec(provider="anthropic", id="claude-opus-5"),
        energy_wh_per_ktok=0.4,
        cached=True,
    )
    assert usage.cost_usd == 0.0 and usage.energy_wh == 0.0


async def test_engine_closes_cleanly(tmp_path):
    engine = Engine(pairwise_config(tmp_path), provider=MockProvider())
    await engine.run()
    await engine.aclose()
