"""Schemas local to the classification stage."""

from pydantic import BaseModel, ConfigDict, field_validator

from science_bot.pipeline.contracts import QuestionClassification


class ClassificationStageInput(BaseModel):
    """Input contract for the classification stage."""

    model_config = ConfigDict(extra="forbid")

    question: str

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        """Validate the question text.

        Args:
            value: Candidate question text.

        Returns:
            str: Stripped question text.

        Raises:
            ValueError: If the question is empty after stripping.
        """
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must be non-empty.")
        return stripped


class ClassificationStageOutput(BaseModel):
    """Output contract for the classification stage."""

    model_config = ConfigDict(extra="forbid")

    classification: QuestionClassification
