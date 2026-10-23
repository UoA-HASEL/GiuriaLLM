"""Provider adapters: request shaping and dataset/CLI plumbing."""

from __future__ import annotations

import json

import pytest

from giuriallm.cli import main
from giuriallm.dataset import DatasetError, load_human_labels, load_items
from giuriallm.providers import LLMRequest, ProviderError
from giuriallm.providers.anthropic_provider import (
    AnthropicProvider,
    price_for,
    rejects_sampling_params,
)
from giuriallm.spec import DatasetFields, DatasetSpec, ModelSpec, ReferenceMode, ThinkingMode

SCHEMA = {"type": "object", "properties": {"best_choice": {"type": "string"}}, "required": []}


def request_for(**model_kwargs) -> LLMRequest:
    return LLMRequest(
        model=ModelSpec(**model_kwargs),
        system="sys",
        user="user",
        schema=SCHEMA,
        tool_name="submit_verdict",
    )


def test_temperature_is_dropped_for_models_that_reject_it():
    provider = AnthropicProvider()
    params = provider.build_params(request_for(id="claude-opus-5", temperature=0.0), strict=True)
    assert "temperature" not in params
    assert any("temperature was dropped" in note for note in provider.notes)


def test_temperature_is_sent_to_models_that_accept_it():
    provider = AnthropicProvider()
    params = provider.build_params(request_for(id="claude-haiku-4-5", temperature=0.0), strict=True)
    assert params["temperature"] == 0.0
    assert provider.notes == []


def test_disabled_thinking_forces_the_verdict_tool():
    params = AnthropicProvider().build_params(
        request_for(id="claude-opus-5", thinking=ThinkingMode.DISABLED), strict=True
    )
    assert params["thinking"] == {"type": "disabled"}
    assert params["tool_choice"] == {"type": "tool", "name": "submit_verdict"}
    assert params["tools"][0]["strict"] is True


def test_adaptive_thinking_does_not_force_tool_choice():
    params = AnthropicProvider().build_params(
        request_for(id="claude-opus-5", thinking=ThinkingMode.ADAPTIVE), strict=True
    )
    assert params["thinking"] == {"type": "adaptive"}
    assert params["tool_choice"] == {"type": "auto"}


def test_disabled_thinking_above_high_effort_is_rejected():
    with pytest.raises(ProviderError, match="effort"):
        AnthropicProvider().build_params(
            request_for(id="claude-opus-5", thinking=ThinkingMode.DISABLED, effort="max"),
            strict=True,
        )


def test_effort_is_passed_through_output_config():
    params = AnthropicProvider().build_params(
        request_for(id="claude-opus-5", effort="high"), strict=True
    )
    assert params["output_config"] == {"effort": "high"}


def test_known_models_have_prices():
    assert price_for(ModelSpec(id="claude-opus-5")) == (5.0, 25.0)
    assert price_for(ModelSpec(id="claude-haiku-4-5")) == (1.0, 5.0)
    assert price_for(ModelSpec(id="some-local-model")) is None
    assert price_for(ModelSpec(id="some-local-model", price_per_mtok=(0.1, 0.2))) == (0.1, 0.2)
    assert rejects_sampling_params("claude-sonnet-5")


def test_cache_key_is_content_addressed():
    a = request_for(id="claude-opus-5")
    b = request_for(id="claude-opus-5")
    c = request_for(id="claude-sonnet-5")
    assert a.cache_key == b.cache_key
    assert a.cache_key != c.cache_key


# -- dataset -------------------------------------------------------------


def spec_for(path, **kwargs) -> DatasetSpec:
    return DatasetSpec(path=path, fields=DatasetFields(**kwargs))


def test_load_items_maps_fields(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text(
        json.dumps({"id": "a", "prompt": "p", "x": "one", "y": "two", "ref": "r", "extra": 7})
        + "\n"
    )
    items = load_items(spec_for(path, candidates=["x", "y"], reference="ref"))
    assert items[0].candidates == ["one", "two"]
    assert items[0].reference == "r"
    assert items[0].meta == {"extra": 7}


def test_missing_candidate_field_is_an_error(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text(json.dumps({"id": "a", "prompt": "p"}) + "\n")
    with pytest.raises(DatasetError, match="no field 'candidate'"):
        load_items(spec_for(path))


def test_required_reference_missing_from_row_is_an_error(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text(json.dumps({"id": "a", "prompt": "p", "candidate": "c"}) + "\n")
    with pytest.raises(DatasetError, match="reference.mode=required"):
        load_items(spec_for(path, reference="ref"), ReferenceMode.REQUIRED)


def test_duplicate_ids_are_rejected(tmp_path):
    path = tmp_path / "d.jsonl"
    row = json.dumps({"id": "a", "prompt": "p", "candidate": "c"})
    path.write_text(row + "\n" + row + "\n")
    with pytest.raises(DatasetError, match="duplicate item id"):
        load_items(spec_for(path))


def test_limit_truncates(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text(
        "\n".join(json.dumps({"id": f"i{i}", "prompt": "p", "candidate": "c"}) for i in range(5))
        + "\n"
    )
    spec = DatasetSpec(path=path, fields=DatasetFields(), limit=2)
    assert len(load_items(spec)) == 2


def test_human_labels_accept_single_and_distribution_forms(tmp_path):
    path = tmp_path / "h.jsonl"
    path.write_text(
        json.dumps({"id": "a", "label": 4})
        + "\n"
        + json.dumps({"id": "b", "labels": [1, 2]})
        + "\n"
    )
    labels = load_human_labels(path)
    assert labels == {"a": [4], "b": [1, 2]}


# -- CLI -----------------------------------------------------------------


def test_cli_validate_reports_hash(tmp_path, capsys):
    config = tmp_path / "c.yaml"
    config.write_text(
        "judge:\n"
        "  type: pointwise\n"
        "  criteria: [{name: c, definition: d}]\n"
        "  outputs: {result: graded}\n"
        "dataset: {path: d.jsonl, fields: {candidates: [candidate]}}\n"
    )
    assert main(["validate", str(config)]) == 0
    assert "config hash" in capsys.readouterr().out


def test_cli_reports_config_errors_without_a_traceback(tmp_path, capsys):
    config = tmp_path / "bad.yaml"
    config.write_text("judge: {type: nonsense}\n")
    assert main(["validate", str(config)]) == 1
    assert "error:" in capsys.readouterr().err


def test_cli_init_then_validate(tmp_path, capsys):
    target = tmp_path / "giuriallm.yaml"
    assert main(["init", str(target)]) == 0
    assert target.is_file()
    assert main(["init", str(target)]) == 1  # refuses to overwrite


def test_cli_run_end_to_end(tmp_path, capsys):
    dataset = tmp_path / "d.jsonl"
    dataset.write_text(json.dumps({"id": "i1", "prompt": "p", "candidate": "c"}) + "\n")
    config = tmp_path / "c.yaml"
    config.write_text(
        "judge:\n"
        "  type: pointwise\n"
        "  criteria: [{name: correctness, definition: 'works?'}]\n"
        "  outputs: {result: graded}\n"
        f"dataset: {{path: {dataset}, fields: {{candidates: [candidate]}}}}\n"
        f"engine: {{cache_dir: {tmp_path / 'cache'}}}\n"
        f"report: {{output_dir: {tmp_path / 'runs'}}}\n"
    )
    assert main(["run", str(config), "--provider", "mock"]) == 0
    out = capsys.readouterr().out
    assert "Items: 1" in out
    assert "provider=mock" in out

    run_dirs = list((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 1
    assert main(["show", str(run_dirs[0])]) == 0
