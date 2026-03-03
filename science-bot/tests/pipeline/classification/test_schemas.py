import pytest
from pydantic import ValidationError
from science_bot.pipeline.classification.schemas import (
    ClassificationStageInput,
    ClassificationStageOutput,
)
from science_bot.pipeline.contracts import (
    SupportedQuestionClassification,
    UnsupportedQuestionClassification,
)


def test_classification_stage_input_accepts_non_empty_question() -> None:
    stage_input = ClassificationStageInput(question="  What is the mean age?  ")

    assert stage_input.question == "What is the mean age?"


def test_classification_stage_input_rejects_empty_question() -> None:
    with pytest.raises(ValidationError, match="question must be non-empty"):
        ClassificationStageInput(question="   ")


def test_classification_stage_output_accepts_supported_classification() -> None:
    output = ClassificationStageOutput(
        classification=SupportedQuestionClassification(family="aggregate")
    )

    assert output.classification.family == "aggregate"


def test_classification_stage_output_accepts_unsupported_classification() -> None:
    output = ClassificationStageOutput(
        classification=UnsupportedQuestionClassification(
            family="unsupported",
            reason="Ambiguous between regression and hypothesis_test.",
        )
    )

    assert output.classification.family == "unsupported"
