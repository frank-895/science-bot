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
from science_bot.tracing import TraceWriter


class OrchestratorRequest(BaseModel):
    """Validated request for running the pipeline orchestrator."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    question: str
    capsule_path: Path
    trace_writer: "TraceWriter | None" = None

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

    if request.trace_writer is not None:
        request.trace_writer.write_event(
            event="classification_started",
            stage="classification",
            question=request.question,
            payload={"capsule_path": request.capsule_path},
        )

    try:
        classification_output = await run_classification_stage(
            ClassificationStageInput(
                question=request.question,
                trace_writer=request.trace_writer,
            )
        )
    except Exception as exc:
        if request.trace_writer is not None:
            request.trace_writer.write_event(
                event="run_failed",
                stage="classification",
                question=request.question,
                payload={"error": str(exc)},
            )
        raise
    classification = classification_output.classification
    if request.trace_writer is not None:
        request.trace_writer.write_event(
            event="classification_finished",
            stage="classification",
            question=request.question,
            family=classification.family,
            payload={
                "classification": classification.model_dump(mode="python"),
            },
        )
    if isinstance(classification, UnsupportedQuestionClassification):
        if request.trace_writer is not None:
            request.trace_writer.write_event(
                event="run_failed",
                stage="classification",
                question=request.question,
                family=classification.family,
                payload={"error": classification.reason},
            )
        raise ValueError(f"Unsupported question: {classification.reason}")

    if request.trace_writer is not None:
        request.trace_writer.write_event(
            event="resolution_started",
            stage="resolution",
            question=request.question,
            family=classification.family,
            payload={"capsule_path": request.capsule_path},
        )
    try:
        resolution_output = await run_resolution_stage(
            ResolutionStageInput(
                question=request.question,
                classification=classification,
                capsule_path=request.capsule_path,
                trace_writer=request.trace_writer,
            )
        )
    except Exception as exc:
        if request.trace_writer is not None:
            request.trace_writer.write_event(
                event="run_failed",
                stage="resolution",
                question=request.question,
                family=classification.family,
                payload={"error": str(exc)},
            )
        raise
    if request.trace_writer is not None:
        request.trace_writer.write_event(
            event="resolution_finished",
            stage="resolution",
            question=request.question,
            family=classification.family,
            payload={
                "iterations_used": resolution_output.iterations_used,
                "selected_files": resolution_output.selected_files,
                "notes": resolution_output.notes,
                "steps": [
                    step.model_dump(mode="python") for step in resolution_output.steps
                ],
            },
        )

    if request.trace_writer is not None:
        request.trace_writer.write_event(
            event="execution_started",
            stage="execution",
            question=request.question,
            family=classification.family,
            payload={
                "payload_family": resolution_output.payload.family,
            },
        )
    try:
        execution_output = run_execution_stage(
            ExecutionStageInput(payload=resolution_output.payload)
        )
    except Exception as exc:
        if request.trace_writer is not None:
            request.trace_writer.write_event(
                event="run_failed",
                stage="execution",
                question=request.question,
                family=classification.family,
                payload={"error": str(exc)},
            )
        raise
    if request.trace_writer is not None:
        request.trace_writer.write_event(
            event="execution_finished",
            stage="execution",
            question=request.question,
            family=classification.family,
            payload={
                "family": execution_output.family,
                "answer": execution_output.answer,
                "raw_result": execution_output.raw_result,
                "notes": execution_output.notes,
            },
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
