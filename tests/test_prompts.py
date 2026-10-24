"""Prompt compilation and verdict parsing, including the failure paths."""

from __future__ import annotations

import pytest

from giuriallm.prompts import PromptCompiler, VerdictParseError, parse_verdict
from giuriallm.spec import (
    Criterion,
    JudgeSpec,
    Outputs,
    PromptOrder,
    PromptSpec,
    Reference,
    ReferenceMode,
    Scale,
)
from giuriallm.spec.models import EvaluationType, ResultKind, ScaleKind
from giuriallm.types import EvaluationItem, Variant

LIKERT = Criterion(
    name="correctness",
    definition="Does it work?",
    scale=Scale(kind=ScaleKind.LIKERT, min=1, max=5, anchors={"1": "no", "5": "yes"}),
)


def pointwise_judge(**kwargs) -> JudgeSpec:
    return JudgeSpec(
        type=EvaluationType.POINTWISE,
        criteria=[LIKERT],
        outputs=Outputs(result=ResultKind.GRADED, **kwargs),
    )


def pairwise_judge() -> JudgeSpec:
    return JudgeSpec(
        type=EvaluationType.PAIRWISE,
        criteria=[LIKERT],
        outputs=Outputs(result=ResultKind.BEST_CHOICE),
    )


ITEM = EvaluationItem(
    id="i1", context="Write mean()", candidates=["def mean(): ...", "def mean2(): ..."]
)


def test_schema_orders_explanation_before_result():
    compiler = PromptCompiler(pointwise_judge(), PromptSpec(order=PromptOrder.EXPLAIN_THEN_RATE))
    keys = list(compiler.schema(1)["properties"])
    assert keys.index("explanation") < keys.index("scores")


def test_schema_orders_result_before_explanation():
    compiler = PromptCompiler(pointwise_judge(), PromptSpec(order=PromptOrder.RATE_THEN_EXPLAIN))
    keys = list(compiler.schema(1)["properties"])
    assert keys.index("scores") < keys.index("explanation")


def test_schema_is_closed_and_requires_every_output():
    judge = pointwise_judge(feedback=True)
    schema = PromptCompiler(judge, PromptSpec()).schema(1)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"explanation", "scores", "feedback"}


def test_pairwise_schema_enumerates_candidate_labels():
    schema = PromptCompiler(pairwise_judge(), PromptSpec()).schema(2)
    assert schema["properties"]["best_choice"]["enum"] == ["A", "B"]


@pytest.mark.parametrize("explanation", [False, True])
def test_criterion_justification_follows_explanation_switch(explanation):
    schema = PromptCompiler(pointwise_judge(explanation=explanation), PromptSpec()).schema(1)
    entry = schema["properties"]["scores"]["items"]
    assert ("justification" in entry["properties"]) is explanation
    assert ("justification" in entry["required"]) is explanation
    assert set(entry["required"]) == set(entry["properties"])
    assert ("explanation" in schema["properties"]) is explanation


def test_verdict_only_prompt_does_not_request_extra_prose_or_a_specific_transport():
    prompt = PromptCompiler(pointwise_judge(explanation=False), PromptSpec()).compile(
        EvaluationItem(id="i", candidates=["code"]), Variant(order=[0])
    )
    assert "Return only the requested structured verdict" in prompt.system
    assert "before the call" not in prompt.system
    assert "calling the" not in prompt.system


def test_prompt_shows_candidates_in_variant_order():
    compiler = PromptCompiler(pairwise_judge(), PromptSpec())
    swapped = compiler.compile(ITEM, Variant(order=[1, 0], index=1))
    body = swapped.user
    assert body.index("def mean2") < body.index("def mean():")


def test_reference_is_withheld_when_mode_is_none():
    judge = JudgeSpec(
        type=EvaluationType.POINTWISE,
        criteria=[LIKERT],
        reference=Reference(mode=ReferenceMode.NONE),
        outputs=Outputs(result=ResultKind.GRADED),
    )
    item = EvaluationItem(id="i", candidates=["x"], reference="the answer")
    prompt = PromptCompiler(judge, PromptSpec()).compile(item, Variant(order=[0]))
    assert "the answer" not in prompt.user


def test_reference_is_included_when_optional():
    judge = JudgeSpec(
        type=EvaluationType.POINTWISE,
        criteria=[LIKERT],
        reference=Reference(mode=ReferenceMode.OPTIONAL),
        outputs=Outputs(result=ResultKind.GRADED),
    )
    item = EvaluationItem(id="i", candidates=["x"], reference="the answer")
    prompt = PromptCompiler(judge, PromptSpec()).compile(item, Variant(order=[0]))
    assert "the answer" in prompt.user


def test_prompt_hash_tracks_content():
    compiler = PromptCompiler(pairwise_judge(), PromptSpec())
    a = compiler.compile(ITEM, Variant(order=[0, 1]))
    b = compiler.compile(ITEM, Variant(order=[1, 0], index=1))
    assert a.hash != b.hash
    assert a.hash == compiler.compile(ITEM, Variant(order=[0, 1])).hash


def test_evidence_is_rendered_into_the_prompt():
    compiler = PromptCompiler(pairwise_judge(), PromptSpec())
    prompt = compiler.compile(ITEM, Variant(order=[0, 1]), {"python_syntax": [{"parses": False}]})
    assert "python_syntax" in prompt.user
    assert "Evidence" in prompt.user


# -- parsing -------------------------------------------------------------


def test_parse_graded_verdict():
    verdict = parse_verdict(
        {"explanation": "looks right", "scores": [{"criterion": "correctness", "score": 4}]},
        pointwise_judge(),
        Variant(order=[0]),
        ["A"],
    )
    assert verdict.score_for("correctness") == 4.0
    assert verdict.explanation == "looks right"


def test_parse_maps_choice_back_through_the_swap():
    """Under order [1, 0], choosing 'A' means the original candidate 1 won."""
    verdict = parse_verdict(
        {"explanation": "b is better", "best_choice": "A"},
        pairwise_judge(),
        Variant(order=[1, 0], index=1),
        ["A", "B"],
    )
    assert verdict.best_choice == 1


def test_parse_rejects_out_of_range_score():
    with pytest.raises(VerdictParseError, match="outside scale"):
        parse_verdict(
            {"explanation": "x", "scores": [{"criterion": "correctness", "score": 9}]},
            pointwise_judge(),
            Variant(order=[0]),
            ["A"],
        )


def test_parse_rejects_non_integer_likert():
    with pytest.raises(VerdictParseError, match="integer"):
        parse_verdict(
            {"explanation": "x", "scores": [{"criterion": "correctness", "score": 3.5}]},
            pointwise_judge(),
            Variant(order=[0]),
            ["A"],
        )


def test_parse_rejects_unknown_criterion():
    with pytest.raises(VerdictParseError, match="unknown criterion"):
        parse_verdict(
            {"explanation": "x", "scores": [{"criterion": "vibes", "score": 3}]},
            pointwise_judge(),
            Variant(order=[0]),
            ["A"],
        )


def test_parse_rejects_missing_criterion():
    judge = JudgeSpec(
        type=EvaluationType.POINTWISE,
        criteria=[LIKERT, Criterion(name="readability", definition="clear?")],
        outputs=Outputs(result=ResultKind.GRADED),
    )
    with pytest.raises(VerdictParseError, match="no score returned"):
        parse_verdict(
            {"explanation": "x", "scores": [{"criterion": "correctness", "score": 3}]},
            judge,
            Variant(order=[0]),
            ["A"],
        )


def test_parse_rejects_missing_explanation_when_required():
    with pytest.raises(VerdictParseError, match="explanation"):
        parse_verdict(
            {"scores": [{"criterion": "correctness", "score": 3}]},
            pointwise_judge(),
            Variant(order=[0]),
            ["A"],
        )


def test_parse_rejects_invalid_label():
    with pytest.raises(VerdictParseError, match="not one of the candidate labels"):
        parse_verdict(
            {"explanation": "x", "best_choice": "Z"},
            pairwise_judge(),
            Variant(order=[0, 1]),
            ["A", "B"],
        )


def test_parse_rejects_ranking_that_is_not_a_permutation():
    judge = JudgeSpec(
        type=EvaluationType.LISTWISE,
        criteria=[LIKERT],
        outputs=Outputs(result=ResultKind.RANKING),
    )
    with pytest.raises(VerdictParseError, match="not a permutation"):
        parse_verdict(
            {"explanation": "x", "ranking": ["A", "A", "B"]},
            judge,
            Variant(order=[0, 1, 2]),
            ["A", "B", "C"],
        )
