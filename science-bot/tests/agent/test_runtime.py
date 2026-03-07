import asyncio
from pathlib import Path

import pytest
from pydantic import ValidationError
from science_bot.agent.contracts import AgentIterationResponse, AgentStepRecord
from science_bot.agent.runtime import run_agent
from science_bot.agent.summary import summarize_steps
from science_bot.providers.executor import PythonExecutionResult
from science_bot.providers.llm import LLMResponseFormatError
from science_bot.tracing import TraceWriter


def test_iteration_response_requires_python_field() -> None:
    with pytest.raises(ValidationError):
        AgentIterationResponse.model_validate({})

    response = AgentIterationResponse.model_validate({"python": "   "})
    assert response.python_code == "   "


def test_iteration_response_ignores_extra_fields() -> None:
    response = AgentIterationResponse.model_validate(
        {"python": "print(1)", "extra": "ignored"}
    )

    assert response.python_code == "print(1)"


def test_run_agent_completes_from_python_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = [AgentIterationResponse.model_validate({"python": "print('x')"})]

    async def fake_parse_structured(**_: object) -> AgentIterationResponse:
        return responses.pop(0)

    async def fake_run_python(
        script: str,
        *,
        timeout_seconds: int | None = None,
        run_id: str | None = None,
    ) -> PythonExecutionResult:
        del script, timeout_seconds
        assert run_id == "q1-iter-1"
        return PythonExecutionResult(
            status="completed",
            answer="irrelevant",
            error_type=None,
            error_message=None,
            stdout_tail="noise\nFINAL_ANSWER: 42\n",
            stderr_tail="",
            duration_ms=5,
            worker="runner-1",
        )

    monkeypatch.setattr(
        "science_bot.agent.runtime.list_available_python_packages",
        lambda: ["numpy"],
    )
    monkeypatch.setattr(
        "science_bot.agent.runtime.parse_structured",
        fake_parse_structured,
    )
    monkeypatch.setattr("science_bot.agent.runtime.run_python", fake_run_python)

    result = asyncio.run(
        run_agent(
            question="How many rows?",
            capsule_path=Path("/capsules/row1"),
            capsule_manifest="/capsules/row1/file.csv",
            execution_id="q1",
            max_iterations=6,
        )
    )

    assert result.status == "completed"
    assert result.answer == "42"
    assert result.iterations_used == 1
    assert len(result.steps) == 1
    assert result.steps[0].execution_status == "completed"
    assert result.steps[0].proposed_final_answer == "42"


def test_run_agent_fails_after_budget_without_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def fake_parse_structured(**_: object) -> AgentIterationResponse:
        return AgentIterationResponse.model_validate({"python": "print(1)"})

    async def fake_run_python(
        script: str,
        *,
        timeout_seconds: int | None = None,
        run_id: str | None = None,
    ) -> PythonExecutionResult:
        del script, timeout_seconds, run_id
        return PythonExecutionResult(
            status="completed",
            answer="some output",
            error_type=None,
            error_message=None,
            stdout_tail="some output",
            stderr_tail="",
            duration_ms=10,
            worker="runner-1",
        )

    monkeypatch.setattr(
        "science_bot.agent.runtime.list_available_python_packages",
        lambda: ["numpy"],
    )
    monkeypatch.setattr(
        "science_bot.agent.runtime.parse_structured",
        fake_parse_structured,
    )
    monkeypatch.setattr("science_bot.agent.runtime.run_python", fake_run_python)

    result = asyncio.run(
        run_agent(
            question="Question?",
            capsule_path=tmp_path,
            capsule_manifest="/capsules/row1/file.csv",
            max_iterations=2,
        )
    )

    assert result.status == "failed"
    assert result.failure_reason == "max_iterations_no_answer"
    assert result.failure_detail is not None
    assert "last_had_python=True" in result.failure_detail
    assert "last_execution_status=completed" in result.failure_detail


def test_run_agent_fails_on_invalid_output_without_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def fake_parse_structured(**_: object) -> AgentIterationResponse:
        raise LLMResponseFormatError("invalid json")

    async def fake_run_python(
        script: str,
        *,
        timeout_seconds: int | None = None,
        run_id: str | None = None,
    ) -> PythonExecutionResult:
        del script, timeout_seconds, run_id
        return PythonExecutionResult(
            status="completed",
            answer="",
            error_type=None,
            error_message=None,
            stdout_tail="FINAL_ANSWER: 42",
            stderr_tail="",
            duration_ms=2,
            worker="runner-1",
        )

    monkeypatch.setattr(
        "science_bot.agent.runtime.list_available_python_packages",
        lambda: ["numpy"],
    )
    monkeypatch.setattr(
        "science_bot.agent.runtime.parse_structured",
        fake_parse_structured,
    )
    monkeypatch.setattr("science_bot.agent.runtime.run_python", fake_run_python)

    result = asyncio.run(
        run_agent(
            question="Question?",
            capsule_path=tmp_path,
            capsule_manifest="/capsules/row1/file.csv",
            max_iterations=1,
        )
    )

    assert result.status == "failed"
    assert result.failure_reason == "invalid_decision_output"
    assert result.failure_detail is not None
    assert "invalid json" in result.failure_detail


def test_run_agent_writes_iteration_trace_events(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    responses = [AgentIterationResponse.model_validate({"python": "print('ok')"})]

    async def fake_parse_structured(**_: object) -> AgentIterationResponse:
        return responses.pop(0)

    async def fake_run_python(
        script: str,
        *,
        timeout_seconds: int | None = None,
        run_id: str | None = None,
    ) -> PythonExecutionResult:
        del script, timeout_seconds, run_id
        return PythonExecutionResult(
            status="completed",
            answer="",
            error_type=None,
            error_message=None,
            stdout_tail="FINAL_ANSWER: done",
            stderr_tail="",
            duration_ms=7,
            worker="runner-1",
        )

    monkeypatch.setattr(
        "science_bot.agent.runtime.list_available_python_packages",
        lambda: ["numpy"],
    )
    monkeypatch.setattr(
        "science_bot.agent.runtime.parse_structured",
        fake_parse_structured,
    )
    monkeypatch.setattr("science_bot.agent.runtime.run_python", fake_run_python)
    trace_writer = TraceWriter(tmp_path / "agent-traces")

    result = asyncio.run(
        run_agent(
            question="Question?",
            capsule_path=tmp_path,
            capsule_manifest="/capsules/row1/file.csv",
            execution_id="trace-q",
            trace_writer=trace_writer,
            max_iterations=2,
        )
    )

    assert result.status == "completed"
    assert result.answer == "done"
    events = (trace_writer.root_dir / "events.jsonl").read_text(encoding="utf-8")
    assert "agent_iteration_started" in events
    assert "agent_decision" in events
    assert "agent_execution_result" in events
    assert "completed_from_python_output" in events


def test_summarize_steps_includes_execution_output_fields() -> None:
    summary = summarize_steps(
        [
            AgentStepRecord(
                iteration=1,
                script="print(1)",
                execution_status="completed",
                execution_answer="answer-value",
                execution_stdout_tail="stdout-value",
                execution_stderr_tail="stderr-value",
                proposed_final_answer="candidate",
            )
        ]
    )

    assert "iter=1" in summary
    assert "exec_status=completed" in summary
    assert "exec_answer=answer-value" in summary
    assert "exec_stdout=stdout-value" in summary
    assert "exec_stderr=stderr-value" in summary
    assert "proposed_final=candidate" in summary
