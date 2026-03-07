"""Pydantic contracts for the agent runtime."""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints, model_validator

NonEmptyStr = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class AgentDecision(BaseModel):
    """One model decision for an agent iteration."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["run_python", "respond", "need_info"]
    script: str | None = None
    answer: str | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def validate_fields_for_decision(self) -> "AgentDecision":
        """Validate required fields for the selected decision.

        Returns:
            AgentDecision: Validated decision.

        Raises:
            ValueError: If required fields are missing.
        """
        if self.decision == "run_python":
            if self.script is None or not self.script.strip():
                raise ValueError("run_python decisions must include non-empty script.")
            if self.answer is not None or self.reason is not None:
                raise ValueError(
                    "run_python decisions cannot include answer or reason."
                )
        if self.decision == "respond":
            if self.answer is None or not self.answer.strip():
                raise ValueError("respond decisions must include non-empty answer.")
            if self.script is not None or self.reason is not None:
                raise ValueError("respond decisions cannot include script or reason.")
        if self.decision == "need_info":
            if self.reason is None or not self.reason.strip():
                raise ValueError("need_info decisions must include a reason.")
            if self.script is not None or self.answer is not None:
                raise ValueError("need_info decisions cannot include script or answer.")
        return self


class AgentStepRecord(BaseModel):
    """One recorded step of the agent runtime."""

    model_config = ConfigDict(extra="forbid")

    iteration: int
    decision: Literal["run_python", "respond", "need_info"]
    reason: str | None = None
    script: str | None = None
    answer: str | None = None
    execution_status: str | None = None
    execution_error: str | None = None


class AgentRunResult(BaseModel):
    """Terminal result returned by the agent runtime."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["completed", "failed"]
    answer: str | None = None
    iterations_used: int
    steps: list[AgentStepRecord]
    failure_reason: NonEmptyStr | None = None


class AgentRunRequest(BaseModel):
    """Validated request for the iterative agent runtime.

    Attributes:
        question: User question to answer.
        capsule_path: Filesystem path to the capsule data directory.
        max_iterations: Maximum number of model decisions to execute.
    """

    model_config = ConfigDict(extra="forbid")

    question: NonEmptyStr
    capsule_path: Path
    max_iterations: int = 6

    @model_validator(mode="after")
    def validate_max_iterations(self) -> "AgentRunRequest":
        """Validate the iteration budget.

        Returns:
            AgentRunRequest: Validated request.

        Raises:
            ValueError: If the iteration budget is invalid.
        """
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be greater than zero.")
        return self
