"""Async pipeline orchestrator entrypoint."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from science_bot.pipeline.classification import (
    ClassificationStageInput,
    run_classification_stage,
)
from science_bot.pipeline.contracts import UnsupportedQuestionClassification
from science_bot.pipeline.execution import ExecutionStageInput, run_execution_stage
from science_bot.pipeline.resolution import ResolutionStageInput, run_resolution_stage


class OrchestratorRequest(BaseModel):
    """Validated request for running the pipeline orchestrator."""

    model_config = ConfigDict(extra="forbid")

    question: str
    capsule_path: Path

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


class OrchestratorResult(BaseModel):
    """Terminal-facing result from the orchestrator."""

    model_config = ConfigDict(extra="forbid")

    question: str
    capsule_path: Path
    status: Literal["completed"]
    answer: str
    metadata: dict[str, object]
    error: str | None = None


async def run_orchestrator(request: OrchestratorRequest) -> OrchestratorResult:
    """Run the pipeline orchestrator.

    Args:
        request: Validated orchestrator request.

    Returns:
        OrchestratorResult: Terminal-facing pipeline result.

    Raises:
        FileNotFoundError: If the requested capsule path does not exist.
        ValueError: If the question is unsupported.
    """
    if not request.capsule_path.exists():
        raise FileNotFoundError(f"Capsule path not found: {request.capsule_path}")

    classification_output = await run_classification_stage(
        ClassificationStageInput(question=request.question)
    )
    classification = classification_output.classification
    if isinstance(classification, UnsupportedQuestionClassification):
        raise ValueError(f"Unsupported question: {classification.reason}")

    resolution_output = await run_resolution_stage(
        ResolutionStageInput(
            question=request.question,
            classification=classification,
            capsule_path=request.capsule_path,
        )
    )
    execution_output = run_execution_stage(
        ExecutionStageInput(payload=resolution_output.payload)
    )

    return OrchestratorResult(
        question=request.question,
        capsule_path=request.capsule_path,
        status="completed",
        answer=execution_output.answer,
        metadata={
            "classification_family": classification.family,
            "resolution_iterations_used": resolution_output.iterations_used,
            "resolution_selected_files": resolution_output.selected_files,
            "resolution_notes": resolution_output.notes,
            "resolution_steps": [
                step.model_dump(mode="python") for step in resolution_output.steps
            ],
            "execution_family": execution_output.family,
            "execution_raw_result": execution_output.raw_result,
            "execution_notes": execution_output.notes,
        },
        error=None,
    )
