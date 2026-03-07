"""Pydantic contracts for the agent runtime."""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
    model_validator,
)

NonEmptyStr = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class RunPythonDecision(BaseModel):
    """Decision variant for executing one Python script."""

    model_config = ConfigDict(extra="ignore")

    decision: Literal["run_python"]
    script: NonEmptyStr


class RespondDecision(BaseModel):
    """Decision variant for returning a candidate answer."""

    model_config = ConfigDict(extra="ignore")

    decision: Literal["respond"]
    answer: NonEmptyStr


class NeedInfoDecision(BaseModel):
    """Decision variant for signaling missing required information."""

    model_config = ConfigDict(extra="ignore")

    decision: Literal["need_info"]
    reason: NonEmptyStr


DecisionPayload = Annotated[
    RunPythonDecision | RespondDecision | NeedInfoDecision,
    Field(discriminator="decision"),
]
_DECISION_ADAPTER = TypeAdapter(DecisionPayload)


class AgentDecision(BaseModel):
    """Decision model validated by a discriminated union adapter."""

    model_config = ConfigDict(extra="ignore")

    decision: Literal["run_python", "respond", "need_info"]
    script: NonEmptyStr | None = None
    answer: NonEmptyStr | None = None
    reason: NonEmptyStr | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_and_validate_payload(cls, data: object) -> object:
        """Validate through the discriminated union and drop irrelevant keys.

        Args:
            data: Candidate decision payload.

        Returns:
            object: Normalized payload.
        """
        if not isinstance(data, dict):
            return data

        payload = _DECISION_ADAPTER.validate_python(data)
        if isinstance(payload, RunPythonDecision):
            return {"decision": "run_python", "script": payload.script}
        if isinstance(payload, RespondDecision):
            return {"decision": "respond", "answer": payload.answer}
        return {"decision": "need_info", "reason": payload.reason}


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
    execution_answer: str | None = None
    execution_stdout_tail: str | None = None
    execution_stderr_tail: str | None = None
    execution_duration_ms: int | None = None
    execution_worker: str | None = None


class AgentRunResult(BaseModel):
    """Terminal result returned by the agent runtime."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["completed", "failed"]
    answer: str | None = None
    iterations_used: int
    steps: list[AgentStepRecord]
    failure_reason: NonEmptyStr | None = None
    failure_detail: str | None = None


class AgentRunRequest(BaseModel):
    """Validated request for the iterative agent runtime.

    Attributes:
        question: User question to answer.
        capsule_path: Filesystem path used by generated Python scripts.
        capsule_manifest: Optional precomputed recursive file listing.
        max_iterations: Maximum number of model decisions to execute.
    """

    model_config = ConfigDict(extra="forbid")

    question: NonEmptyStr
    capsule_path: Path
    capsule_manifest: str | None = None
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
