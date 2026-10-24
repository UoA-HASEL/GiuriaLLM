"""Config schema: round-trip, hashing, and validation failures.

A malformed config must fail loudly with an actionable message, never default
silently.
"""

from __future__ import annotations

import pytest

from giuriallm.spec import (
    ConfigError,
    RunConfig,
    config_hash,
    dump_config,
    load_config,
    parse_config,
)

MINIMAL = {
    "judge": {
        "type": "pointwise",
        "criteria": [{"name": "correctness", "definition": "Does it work?"}],
        "outputs": {"result": "graded"},
    },
    "dataset": {"path": "data/items.jsonl", "fields": {"candidates": ["candidate"]}},
}


def test_minimal_config_validates():
    config = parse_config(MINIMAL)
    assert config.judge.type.value == "pointwise"
    assert config.model.id == "claude-opus-5"
    assert config.judge.criteria[0].scale.min == 1


def test_round_trip_preserves_semantics():
    config = parse_config(MINIMAL)
    again = parse_config(__import__("yaml").safe_load(dump_config(config)))
    assert config_hash(config) == config_hash(again)


def test_hash_is_stable_under_key_order():
    reordered = {"dataset": MINIMAL["dataset"], "judge": MINIMAL["judge"]}
    assert config_hash(parse_config(MINIMAL)) == config_hash(parse_config(reordered))


def test_hash_ignores_non_semantic_fields():
    a = parse_config(MINIMAL)
    b = parse_config({**MINIMAL, "name": "run-b", "engine": {"concurrency": 32}})
    assert config_hash(a) == config_hash(b)


def test_hash_changes_with_criteria():
    changed = {
        **MINIMAL,
        "judge": {
            **MINIMAL["judge"],
            "criteria": [{"name": "correctness", "definition": "Different wording."}],
        },
    }
    assert config_hash(parse_config(MINIMAL)) != config_hash(parse_config(changed))


def test_unknown_key_is_rejected():
    with pytest.raises(ConfigError) as exc:
        parse_config({**MINIMAL, "judgge": {}})
    assert "judgge" in str(exc.value)


def test_pairwise_requires_two_candidate_fields():
    with pytest.raises(ConfigError) as exc:
        parse_config(
            {
                "judge": {
                    "type": "pairwise",
                    "criteria": MINIMAL["judge"]["criteria"],
                    "outputs": {"result": "best_choice"},
                },
                "dataset": {"path": "d.jsonl", "fields": {"candidates": ["only_one"]}},
            }
        )
    assert "exactly 2" in str(exc.value)


def test_result_kind_must_match_evaluation_type():
    with pytest.raises(ConfigError) as exc:
        parse_config({**MINIMAL, "judge": {**MINIMAL["judge"], "outputs": {"result": "ranking"}}})
    assert "not valid for judge.type=pointwise" in str(exc.value)


def test_position_swap_rejected_for_pointwise():
    with pytest.raises(ConfigError) as exc:
        parse_config({**MINIMAL, "debias": {"position_swap": True}})
    assert "pairwise" in str(exc.value)


def test_likert_scale_requires_bounds():
    with pytest.raises(ConfigError) as exc:
        parse_config(
            {
                **MINIMAL,
                "judge": {
                    **MINIMAL["judge"],
                    "criteria": [
                        {"name": "c", "definition": "d", "scale": {"kind": "likert", "min": 1}}
                    ],
                },
            }
        )
    assert "requires 'min' and 'max'" in str(exc.value)


def test_required_reference_needs_dataset_field():
    with pytest.raises(ConfigError) as exc:
        parse_config({**MINIMAL, "judge": {**MINIMAL["judge"], "reference": {"mode": "required"}}})
    assert "dataset.fields.reference" in str(exc.value)


def test_duplicate_criteria_rejected():
    with pytest.raises(ConfigError):
        parse_config(
            {
                **MINIMAL,
                "judge": {
                    **MINIMAL["judge"],
                    "criteria": [
                        {"name": "c", "definition": "one"},
                        {"name": "c", "definition": "two"},
                    ],
                },
            }
        )


def test_missing_file_reports_path(tmp_path):
    with pytest.raises(ConfigError) as exc:
        load_config(tmp_path / "nope.yaml")
    assert "not found" in str(exc.value)


def test_paths_resolve_relative_to_the_config(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "config.yaml").write_text(
        "judge:\n"
        "  type: pointwise\n"
        "  criteria: [{name: c, definition: d}]\n"
        "  outputs: {result: graded}\n"
        "dataset:\n"
        "  path: data/items.jsonl\n"
        "  fields: {candidates: [candidate]}\n"
    )
    config = load_config(tmp_path / "config.yaml")
    assert config.dataset.path == (tmp_path / "data" / "items.jsonl").resolve()


def test_judges_returns_panel_when_present():
    config = parse_config(
        {
            **MINIMAL,
            "panel": {
                "members": [{"id": "claude-opus-5"}, {"id": "claude-sonnet-5"}],
                "aggregation": "mean",
            },
        }
    )
    assert [m.id for m in config.judges()] == ["claude-opus-5", "claude-sonnet-5"]
    assert isinstance(config, RunConfig)


def test_empty_panel_is_rejected():
    with pytest.raises(ConfigError) as exc:
        parse_config({**MINIMAL, "panel": {"members": []}})
    assert "omit 'panel'" in str(exc.value)
