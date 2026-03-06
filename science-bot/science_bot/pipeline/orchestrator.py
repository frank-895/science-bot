"""Async pipeline orchestrator entrypoint."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from science_bot.tracing import TraceWriter


class OrchestratorRequest(BaseModel):
    """Validated request for running the pipeline orchestrator.

    Attributes:
        question: User question that should be answered from the capsule.
        capsule_path: Filesystem path to the extracted capsule directory.
        trace_writer: Optional trace writer for structured run diagnostics.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    question: str
    capsule_path: Path
    trace_writer: TraceWriter | None = None

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        """Validate question text.

        Args:
            value: Candidate question text.

        Returns:
            str: Stripped question text.

        Raises:
            ValueError: If the question is empty after stripping.
        """
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must be non-empty")
        return stripped


class OrchestratorResult(BaseModel):
    """Terminal-facing result from the orchestrator.

    Attributes:
        question: Original user question.
        capsule_path: Filesystem path used for execution.
        status: Terminal status for run completion.
        answer: Final answer string returned to CLI callers.
        metadata: Structured metadata payload for downstream formatting.
        error: Optional error message when failure details are attached.
    """

    model_config = ConfigDict(extra="forbid")

    question: str
    capsule_path: Path
    status: Literal["completed"]
    answer: str
    metadata: dict[str, object]
    error: str | None = None


async def run_orchestrator(request: OrchestratorRequest) -> OrchestratorResult:
    """Run a temporary stub orchestrator.

    Args:
        request: Validated orchestrator request.

    Returns:
        OrchestratorResult: Deterministic placeholder output.

    Raises:
        FileNotFoundError: If the capsule path does not exist.
    """
    if not request.capsule_path.exists():
        raise FileNotFoundError(f"Capsule path not found: {request.capsule_path}")

    if request.trace_writer is not None:
        request.trace_writer.write_event(
            event="orchestrator_stub_used",
            stage="orchestrator",
            question=request.question,
            payload={"capsule_path": request.capsule_path},
        )

    return OrchestratorResult(
        question=request.question,
        capsule_path=request.capsule_path,
        status="completed",
        answer="ORCHESTRATOR_STUB_RESPONSE",
        metadata={
            "classification_family": "stub",
            "resolution_iterations_used": 0,
            "resolution_selected_files": [],
            "execution_family": "stub",
            "execution_notes": ["Stub orchestrator: analysis not yet implemented."],
        },
        error=None,
    )
