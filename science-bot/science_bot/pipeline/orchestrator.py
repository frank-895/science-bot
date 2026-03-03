"""Async pipeline orchestrator entrypoint."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

STUB_ANSWER = "ORCHESTRATOR_STUB_RESPONSE"


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
    metadata: dict[str, str | int | float | bool | None]
    error: str | None = None


async def run_orchestrator(request: OrchestratorRequest) -> OrchestratorResult:
    """Run the pipeline orchestrator.

    Args:
        request: Validated orchestrator request.

    Returns:
        OrchestratorResult: Stub result until the pipeline is implemented.

    Raises:
        FileNotFoundError: If the requested capsule path does not exist.
    """
    if not request.capsule_path.exists():
        raise FileNotFoundError(f"Capsule path not found: {request.capsule_path}")

    return OrchestratorResult(
        question=request.question,
        capsule_path=request.capsule_path,
        status="completed",
        answer=STUB_ANSWER,
        metadata={"orchestrator_mode": "stub"},
        error=None,
    )
