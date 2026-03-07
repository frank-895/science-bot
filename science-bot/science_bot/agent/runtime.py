"""Real-mode runtime loop for iterative question answering."""

import json
import re
from pathlib import Path

from pydantic import ValidationError

from science_bot.agent.contracts import (
    AgentIterationResponse,
    AgentRunRequest,
    AgentRunResult,
    AgentStepRecord,
)
from science_bot.agent.prompts import (
    build_system_prompt,
    build_user_prompt,
)
from science_bot.agent.summary import summarize_steps
from science_bot.providers.executor import (
    list_available_python_packages,
    run_python,
)
from science_bot.providers.llm import LLMResponseFormatError, parse_structured
from science_bot.tracing import TraceWriter

DEFAULT_MAX_ITERATIONS = 6
DEFAULT_PYTHON_TIMEOUT_SECONDS = 30
FINAL_ANSWER_MARKER = "FINAL_ANSWER:"


async def run_agent(
    *,
    question: str,
    capsule_path: Path,
    capsule_manifest: str | None = None,
    execution_id: str | None = None,
    trace_writer: TraceWriter | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> AgentRunResult:
    """Run the real-mode agent loop.

    Args:
        question: Natural language question to answer.
        capsule_path: Capsule path used by generated scripts.
        capsule_manifest: Optional precomputed recursive file listing.
        execution_id: Optional question-scoped execution identifier.
        trace_writer: Optional trace writer for iteration-level diagnostics.
        max_iterations: Maximum number of decision iterations.

    Returns:
        AgentRunResult: Terminal runtime result with all recorded steps.

    Raises:
        PythonExecutorUnavailableError: If package discovery or execution fails.
    """
    request = AgentRunRequest(
        question=question,
        capsule_path=capsule_path,
        capsule_manifest=capsule_manifest,
        max_iterations=max_iterations,
    )
    available_packages = _safe_list_packages()
    prompt_capsule_path = request.capsule_path
    resolved_manifest = request.capsule_manifest or "(manifest unavailable)"
    steps: list[AgentStepRecord] = []
    system_prompt = build_system_prompt(request.max_iterations)

    for iteration in range(1, request.max_iterations + 1):
        remaining = request.max_iterations - iteration + 1
        _write_trace_event(
            trace_writer=trace_writer,
            event="agent_iteration_started",
            payload={
                "iteration": iteration,
                "remaining_budget": remaining,
                "execution_id": execution_id,
            },
        )
        user_prompt = build_user_prompt(
            question=request.question,
            capsule_path=prompt_capsule_path,
            capsule_manifest=resolved_manifest,
            available_packages=available_packages,
            step_summary=summarize_steps(steps),
            iteration=iteration,
            max_iterations=request.max_iterations,
        )
        try:
            decision = await parse_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=AgentIterationResponse,
                trace_writer=trace_writer,
                trace_stage="agent",
            )
        except (LLMResponseFormatError, ValidationError) as exc:
            failure_detail = f"Decision parsing failed: {exc}"
            _write_trace_event(
                trace_writer=trace_writer,
                event="agent_terminated",
                payload={
                    "iteration": iteration,
                    "reason": "invalid_decision_output",
                    "detail": failure_detail,
                },
            )
            return AgentRunResult(
                status="failed",
                iterations_used=iteration,
                steps=steps,
                failure_reason="invalid_decision_output",
                failure_detail=failure_detail,
            )

        _write_trace_event(
            trace_writer=trace_writer,
            event="agent_decision",
            payload={
                "iteration": iteration,
                "has_python": True,
            },
        )
        run_id = _build_execution_run_id(execution_id, iteration)
        execution_result = await run_python(
            decision.python_code,
            timeout_seconds=DEFAULT_PYTHON_TIMEOUT_SECONDS,
            run_id=run_id,
        )
        step = AgentStepRecord(
            iteration=iteration,
            script=decision.python_code,
            execution_status=execution_result.status,
            execution_error=execution_result.error_message,
            execution_answer=execution_result.answer,
            execution_stdout_tail=execution_result.stdout_tail,
            execution_stderr_tail=execution_result.stderr_tail,
        )
        steps.append(step)
        _write_trace_event(
            trace_writer=trace_writer,
            event="agent_execution_result",
            payload={
                "iteration": iteration,
                "execution_id": execution_id,
                "run_id": run_id,
                "status": execution_result.status,
                "error": execution_result.error_message,
                "duration_ms": execution_result.duration_ms,
                "worker": execution_result.worker,
                "stdout_tail": execution_result.stdout_tail[:240],
                "stderr_tail": execution_result.stderr_tail[:240],
                "answer": execution_result.answer[:200]
                if execution_result.answer
                else None,
            },
        )
        extracted_answer = _extract_final_answer_marker(
            execution_result.answer,
            execution_result.stdout_tail,
        )
        if extracted_answer is not None:
            steps[-1].proposed_final_answer = extracted_answer
            _write_trace_event(
                trace_writer=trace_writer,
                event="agent_terminated",
                payload={
                    "iteration": iteration,
                    "reason": "completed_from_python_output",
                    "answer_preview": extracted_answer[:200],
                },
            )
            return AgentRunResult(
                status="completed",
                answer=extracted_answer,
                iterations_used=iteration,
                steps=steps,
                failure_reason=None,
                failure_detail=None,
            )

    last_step = steps[-1] if steps else None
    no_answer_detail = (
        f"last_had_python={bool(last_step.script.strip())}; "
        f"last_execution_status={last_step.execution_status}"
        if last_step is not None
        else "last_had_python=None; last_execution_status=None"
    )
    _write_trace_event(
        trace_writer=trace_writer,
        event="agent_terminated",
        payload={
            "iteration": request.max_iterations,
            "reason": "max_iterations_no_answer",
            "detail": no_answer_detail,
        },
    )
    return AgentRunResult(
        status="failed",
        iterations_used=request.max_iterations,
        steps=steps,
        failure_reason="max_iterations_no_answer",
        failure_detail=no_answer_detail,
    )


def _safe_list_packages() -> list[str]:
    """Load available execution packages for prompt conditioning.

    Returns:
        list[str]: Sorted package list.

    Raises:
        PythonExecutorUnavailableError: If the executor package cannot be queried.
    """
    packages = list_available_python_packages()
    return sorted(packages)


def _build_execution_run_id(execution_id: str | None, iteration: int) -> str:
    """Build a row-attributed execution run identifier.

    Args:
        execution_id: Optional identifier tied to one benchmark row.
        iteration: Current 1-based iteration.

    Returns:
        str: Stable run identifier used for script artifacts.
    """
    if execution_id is None:
        return f"iter-{iteration}"
    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", execution_id).strip("_")
    if not cleaned:
        cleaned = "row"
    return f"{cleaned}-iter-{iteration}"


def _write_trace_event(
    *,
    trace_writer: TraceWriter | None,
    event: str,
    payload: dict[str, object],
) -> None:
    """Write one optional runtime trace event.

    Args:
        trace_writer: Optional trace writer.
        event: Stable event name.
        payload: Event payload mapping.
    """
    if trace_writer is None:
        return
    trace_writer.write_event(event=event, stage="agent", payload=payload)


def _extract_final_answer_marker(
    answer: str | None,
    stdout_tail: str | None,
) -> str | None:
    """Extract a FINAL_ANSWER marker from execution output.

    Args:
        answer: Structured execution answer field.
        stdout_tail: Tail of raw stdout text.

    Returns:
        str | None: Extracted final answer text if marker is present.
    """
    answer_extracted = _extract_marker_from_text(answer)
    if answer_extracted is not None:
        return answer_extracted

    if stdout_tail is None:
        return None

    # Runner stdout is JSON-encoded. Prefer the parsed "answer" field if present.
    try:
        payload = json.loads(stdout_tail)
    except Exception:
        payload = None
    if isinstance(payload, dict):
        payload_answer = payload.get("answer")
        if isinstance(payload_answer, str):
            payload_extracted = _extract_marker_from_text(payload_answer)
            if payload_extracted is not None:
                return payload_extracted

    return _extract_marker_from_text(stdout_tail)


def _extract_marker_from_text(text: str | None) -> str | None:
    """Extract FINAL_ANSWER marker from plain text lines.

    Args:
        text: Candidate plain text.

    Returns:
        str | None: Marker value if present.
    """
    if text is None:
        return None
    for line in text.splitlines():
        if FINAL_ANSWER_MARKER in line:
            _, _, value = line.partition(FINAL_ANSWER_MARKER)
            extracted = value.strip()
            if extracted:
                return extracted
    return None
