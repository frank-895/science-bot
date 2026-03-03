import asyncio

import pytest
from pydantic import ValidationError
from science_bot.pipeline.classification import stage
from science_bot.pipeline.classification.schemas import ClassificationStageInput
from science_bot.pipeline.contracts import UnsupportedQuestionClassification
from science_bot.providers.llm import LLMProviderError


def test_run_classification_stage_returns_aggregate_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_parse_structured(**_: object) -> stage.ClassificationResponse:
        return stage.ClassificationResponse(family="aggregate")

    monkeypatch.setattr(stage, "parse_structured", fake_parse_structured)

    result = asyncio.run(
        stage.run_classification_stage(
            ClassificationStageInput(question="What is the mean expression?")
        )
    )

    assert result.classification.family == "aggregate"


def test_run_classification_stage_returns_regression_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_parse_structured(**_: object) -> stage.ClassificationResponse:
        return stage.ClassificationResponse(family="regression")

    monkeypatch.setattr(stage, "parse_structured", fake_parse_structured)

    result = asyncio.run(
        stage.run_classification_stage(
            ClassificationStageInput(
                question="What is the odds ratio in a logistic regression model?"
            )
        )
    )

    assert result.classification.family == "regression"


def test_run_classification_stage_returns_differential_expression_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_parse_structured(**_: object) -> stage.ClassificationResponse:
        return stage.ClassificationResponse(family="differential_expression")

    monkeypatch.setattr(stage, "parse_structured", fake_parse_structured)

    result = asyncio.run(
        stage.run_classification_stage(
            ClassificationStageInput(
                question=(
                    "How many significantly differentially expressed genes are there?"
                )
            )
        )
    )

    assert result.classification.family == "differential_expression"


def test_run_classification_stage_returns_unsupported_for_out_of_scope_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_parse_structured(**_: object) -> stage.ClassificationResponse:
        return stage.ClassificationResponse(
            family="unsupported",
            reason=(
                "Question asks for a literature summary, not a supported "
                "analysis family."
            ),
        )

    monkeypatch.setattr(stage, "parse_structured", fake_parse_structured)

    result = asyncio.run(
        stage.run_classification_stage(
            ClassificationStageInput(question="Summarize the background of this paper.")
        )
    )

    assert result.classification.family == "unsupported"
    assert isinstance(result.classification, UnsupportedQuestionClassification)
    assert result.classification.reason.startswith("Question asks")


def test_run_classification_stage_returns_unsupported_for_ambiguous_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_parse_structured(**_: object) -> stage.ClassificationResponse:
        return stage.ClassificationResponse(
            family="unsupported",
            reason="Question could refer to either aggregate or hypothesis_test.",
        )

    monkeypatch.setattr(stage, "parse_structured", fake_parse_structured)

    result = asyncio.run(
        stage.run_classification_stage(
            ClassificationStageInput(question="How different are the groups?")
        )
    )

    assert result.classification.family == "unsupported"
    assert isinstance(result.classification, UnsupportedQuestionClassification)
    assert "either aggregate or hypothesis_test" in result.classification.reason


def test_run_classification_stage_normalizes_supported_reason_away(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_parse_structured(**_: object) -> stage.ClassificationResponse:
        return stage.ClassificationResponse(
            family="aggregate",
            reason="This stray reason should be dropped.",
        )

    monkeypatch.setattr(stage, "parse_structured", fake_parse_structured)

    result = asyncio.run(
        stage.run_classification_stage(
            ClassificationStageInput(question="How many samples are there?")
        )
    )

    assert result.classification.family == "aggregate"
    assert not hasattr(result.classification, "reason")


def test_run_classification_stage_rejects_unsupported_without_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_parse_structured(**_: object) -> stage.ClassificationResponse:
        return stage.ClassificationResponse(family="unsupported")

    monkeypatch.setattr(stage, "parse_structured", fake_parse_structured)

    with pytest.raises(ValidationError, match="unsupported.reason"):
        asyncio.run(
            stage.run_classification_stage(
                ClassificationStageInput(question="Do something unclear.")
            )
        )


def test_run_classification_stage_propagates_provider_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_parse_structured(**_: object) -> stage.ClassificationResponse:
        raise LLMProviderError("provider down")

    monkeypatch.setattr(stage, "parse_structured", fake_parse_structured)

    with pytest.raises(LLMProviderError, match="provider down"):
        asyncio.run(
            stage.run_classification_stage(
                ClassificationStageInput(question="What is the mean age?")
            )
        )


def test_run_classification_stage_propagates_invalid_provider_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InvalidResponse:
        def model_dump(self, *, exclude_none: bool = False) -> dict[str, object]:
            assert exclude_none is True
            return {"family": "made_up_family"}

    async def fake_parse_structured(**_: object) -> InvalidResponse:
        return InvalidResponse()

    monkeypatch.setattr(stage, "parse_structured", fake_parse_structured)

    with pytest.raises(ValidationError):
        asyncio.run(
            stage.run_classification_stage(
                ClassificationStageInput(question="What is the mean age?")
            )
        )
