"""Thin adapter over the standalone executor package."""

import importlib
from typing import Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict

__all__ = [
    "PythonExecutionResult",
    "PythonExecutorUnavailableError",
    "ensure_python_executor_ready",
    "list_available_python_packages",
    "run_python",
]


class PythonExecutorUnavailableError(RuntimeError):
    """Raised when Python execution workers are not ready or install is missing."""


class PythonExecutionResult(BaseModel):
    """Structured result returned from Python script execution."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["completed", "failed", "timeout", "invalid_output"]
    answer: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    duration_ms: int
    worker: str


class _ExecutorApi(Protocol):
    """Protocol for the external executor package API."""

    def ensure_ready(self) -> None:
        """Validate executor readiness."""

    def packages_available(self) -> list[str]:
        """Return packages available in executor workers."""

    async def run_python(
        self,
        script: str,
        *,
        timeout_seconds: int | None = None,
        run_id: str | None = None,
    ) -> dict[str, object]:
        """Execute one python script and return normalized result payload."""


def ensure_python_executor_ready() -> None:
    """Validate Python executor readiness.

    Raises:
        PythonExecutorUnavailableError: If readiness checks fail.
    """
    api = _load_executor_api()
    try:
        api.ensure_ready()
    except Exception as exc:
        raise PythonExecutorUnavailableError(str(exc)) from exc


def list_available_python_packages() -> list[str]:
    """Return deterministic packages available to execution workers.

    Returns:
        list[str]: Package names available inside workers.

    Raises:
        PythonExecutorUnavailableError: If executor package cannot be loaded.
    """
    api = _load_executor_api()
    try:
        return api.packages_available()
    except Exception as exc:
        raise PythonExecutorUnavailableError(str(exc)) from exc


async def run_python(
    script: str,
    *,
    timeout_seconds: int | None = None,
    run_id: str | None = None,
) -> PythonExecutionResult:
    """Execute one Python script through the executor package.

    Args:
        script: Python script source code.
        timeout_seconds: Optional timeout override in seconds.
        run_id: Optional run identifier for artifact grouping.

    Returns:
        PythonExecutionResult: Structured execution outcome.

    Raises:
        PythonExecutorUnavailableError: If execution fails due to runtime issues.
    """
    api = _load_executor_api()
    try:
        payload = await api.run_python(
            script,
            timeout_seconds=timeout_seconds,
            run_id=run_id,
        )
    except Exception as exc:
        raise PythonExecutorUnavailableError(str(exc)) from exc

    try:
        return PythonExecutionResult.model_validate(payload)
    except Exception as exc:
        raise PythonExecutorUnavailableError(str(exc)) from exc


def _load_executor_api() -> _ExecutorApi:
    """Load executor package lazily for explicit editable-install workflows.

    Returns:
        object: Loaded executor package module.

    Raises:
        PythonExecutorUnavailableError: If the package import fails.
    """
    try:
        return cast(_ExecutorApi, importlib.import_module("executor"))
    except ModuleNotFoundError as exc:
        raise PythonExecutorUnavailableError(
            "Executor package is not installed. Run: uv pip install -e ./executor"
        ) from exc
