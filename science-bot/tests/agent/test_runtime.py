import asyncio
from pathlib import Path

import pytest
from pydantic import ValidationError
from science_bot.agent.contracts import AgentDecision, AgentStepRecord
from science_bot.agent.runtime import run_agent
from science_bot.agent.summary import summarize_steps
from science_bot.providers.executor import PythonExecutionResult
from science_bot.providers.llm import LLMResponseFormatError
from science_bot.tracing import TraceWriter


def test_run_agent_records_python_then_respond_steps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    decisions = [
        AgentDecision.model_validate(
            {"decision": "run_python", "script": "print('x')"}
        ),
        AgentDecision.model_validate({"decision": "respond", "answer": "42"}),
        AgentDecision.model_validate({"decision": "respond", "answer": "43"}),
    ]

    async def fake_parse_structured(**_: object) -> AgentDecision:
        return decisions.pop(0)

    async def fake_run_python(
        script: str,
        *,
        timeout_seconds: int | None = None,
        run_id: str | None = None,
    ) -> PythonExecutionResult:
        del timeout_seconds
        assert "print" in script
        assert run_id == "bix-1-iter-1"
        return PythonExecutionResult(
            status="completed",
            answer="value=42",
            error_type=None,
            error_message=None,
            stdout_tail="ok stdout",
            stderr_tail="warn stderr",
            duration_ms=5,
            worker="runner-1",
        )

    monkeypatch.setattr(
        "science_bot.agent.runtime.list_available_python_packages",
        lambda: ["numpy", "pandas"],
    )
    monkeypatch.setattr(
        "science_bot.agent.runtime.parse_structured",
        fake_parse_structured,
    )
    monkeypatch.setattr("science_bot.agent.runtime.run_python", fake_run_python)

    result = asyncio.run(
        run_agent(
            question="How many rows?",
            capsule_path=tmp_path,
            capsule_manifest="/capsules/row1/file.csv",
            execution_id="bix-1",
            max_iterations=3,
        )
    )

    assert result.status == "completed"
    assert result.answer == "43"
    assert result.iterations_used == 3
    assert len(result.steps) == 3
    assert result.steps[0].decision == "run_python"
    assert result.steps[0].execution_status == "completed"
    assert result.steps[0].execution_answer == "value=42"
    assert result.steps[0].execution_stdout_tail == "ok stdout"
    assert result.steps[0].execution_stderr_tail == "warn stderr"
    assert result.steps[0].execution_duration_ms == 5
    assert result.steps[0].execution_worker == "runner-1"
    assert result.steps[2].decision == "respond"
    assert result.steps[2].answer == "43"


def test_run_agent_need_info_is_terminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def fake_parse_structured(**_: object) -> AgentDecision:
        return AgentDecision.model_validate(
            {"decision": "need_info", "reason": "Missing cohort labels"}
        )

    monkeypatch.setattr(
        "science_bot.agent.runtime.list_available_python_packages",
        lambda: ["numpy"],
    )
    monkeypatch.setattr(
        "science_bot.agent.runtime.parse_structured",
        fake_parse_structured,
    )

    result = asyncio.run(
        run_agent(
            question="What is the result?",
            capsule_path=tmp_path,
            capsule_manifest="/capsules/row1/file.csv",
            max_iterations=6,
        )
    )

    assert result.status == "failed"
    assert result.failure_reason == "need_info"
    assert result.iterations_used == 1
    assert len(result.steps) == 1
    assert result.steps[0].decision == "need_info"


def test_run_agent_returns_latest_candidate_after_budget(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    decisions = [
        AgentDecision.model_validate({"decision": "respond", "answer": "candidate-1"}),
        AgentDecision.model_validate(
            {"decision": "run_python", "script": "print('hi')"}
        ),
        AgentDecision.model_validate({"decision": "respond", "answer": "candidate-2"}),
    ]

    async def fake_parse_structured(**_: object) -> AgentDecision:
        return decisions.pop(0)

    async def fake_run_python(
        script: str,
        *,
        timeout_seconds: int | None = None,
        run_id: str | None = None,
    ) -> PythonExecutionResult:
        del script
        del timeout_seconds
        del run_id
        return PythonExecutionResult(
            status="completed",
            answer=None,
            error_type=None,
            error_message=None,
            stdout_tail="",
            stderr_tail="",
            duration_ms=3,
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
            max_iterations=3,
        )
    )

    assert result.status == "completed"
    assert result.answer == "candidate-2"
    assert result.iterations_used == 3


def test_run_agent_fails_when_no_answer_after_budget(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def fake_parse_structured(**_: object) -> AgentDecision:
        return AgentDecision.model_validate(
            {"decision": "run_python", "script": "print(1)"}
        )

    async def fake_run_python(
        script: str,
        *,
        timeout_seconds: int | None = None,
        run_id: str | None = None,
    ) -> PythonExecutionResult:
        del script
        del timeout_seconds
        del run_id
        return PythonExecutionResult(
            status="failed",
            answer=None,
            error_type="runtime_error",
            error_message="boom",
            stdout_tail="",
            stderr_tail="trace",
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
    assert "last_decision=run_python" in result.failure_detail
    assert "last_execution_status=failed" in result.failure_detail
    assert result.iterations_used == 2
    assert len(result.steps) == 2


def test_agent_decision_rejects_invalid_field_combinations() -> None:
    with pytest.raises(ValidationError):
        AgentDecision.model_validate({"decision": "run_python"})
    with pytest.raises(ValidationError):
        AgentDecision.model_validate({"decision": "respond"})
    with pytest.raises(ValidationError):
        AgentDecision.model_validate({"decision": "need_info"})


def test_agent_decision_ignores_harmless_extra_fields() -> None:
    decision = AgentDecision.model_validate(
        {
            "decision": "run_python",
            "script": "print(1)",
            "reason": "extra",
            "answer": "extra",
        }
    )

    assert decision.decision == "run_python"
    assert decision.script == "print(1)"
    assert decision.answer is None
    assert decision.reason is None


def test_run_agent_repairs_invalid_decision_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = {"count": 0}

    async def fake_parse_structured(**_: object) -> AgentDecision:
        calls["count"] += 1
        if calls["count"] == 1:
            raise LLMResponseFormatError("invalid json")
        return AgentDecision.model_validate({"decision": "respond", "answer": "42"})

    monkeypatch.setattr(
        "science_bot.agent.runtime.list_available_python_packages",
        lambda: ["numpy"],
    )
    monkeypatch.setattr(
        "science_bot.agent.runtime.parse_structured",
        fake_parse_structured,
    )

    result = asyncio.run(
        run_agent(
            question="Question?",
            capsule_path=tmp_path,
            capsule_manifest="/capsules/row1/file.csv",
            max_iterations=1,
        )
    )

    assert calls["count"] == 2
    assert result.status == "completed"
    assert result.answer == "42"
    assert result.iterations_used == 1


def test_run_agent_fails_after_repair_retry_exhausted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def fake_parse_structured(**_: object) -> AgentDecision:
        raise LLMResponseFormatError("still invalid")

    monkeypatch.setattr(
        "science_bot.agent.runtime.list_available_python_packages",
        lambda: ["numpy"],
    )
    monkeypatch.setattr(
        "science_bot.agent.runtime.parse_structured",
        fake_parse_structured,
    )

    result = asyncio.run(
        run_agent(
            question="Question?",
            capsule_path=tmp_path,
            capsule_manifest="/capsules/row1/file.csv",
            max_iterations=3,
        )
    )

    assert result.status == "failed"
    assert result.failure_reason == "invalid_decision_output"
    assert result.failure_detail is not None
    assert result.iterations_used == 1


def test_summarize_steps_includes_execution_output_fields() -> None:
    summary = summarize_steps(
        [
            AgentStepRecord(
                iteration=1,
                decision="run_python",
                execution_status="completed",
                execution_answer="answer-value",
                execution_stdout_tail="stdout-value",
                execution_stderr_tail="stderr-value",
                execution_duration_ms=15,
                execution_worker="runner-2",
            )
        ]
    )

    assert "exec_status=completed" in summary
    assert "exec_answer=answer-value" in summary
    assert "exec_stdout=stdout-value" in summary
    assert "exec_stderr=stderr-value" in summary
    assert "exec_ms=15" in summary
    assert "exec_worker=runner-2" in summary


def test_run_agent_writes_iteration_trace_events(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    decisions = [
        AgentDecision.model_validate({"decision": "run_python", "script": "print(1)"}),
        AgentDecision.model_validate({"decision": "respond", "answer": "42"}),
    ]

    async def fake_parse_structured(**_: object) -> AgentDecision:
        return decisions.pop(0)

    async def fake_run_python(
        script: str,
        *,
        timeout_seconds: int | None = None,
        run_id: str | None = None,
    ) -> PythonExecutionResult:
        del script
        del timeout_seconds
        del run_id
        return PythonExecutionResult(
            status="completed",
            answer="42",
            error_type=None,
            error_message=None,
            stdout_tail="42",
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
            execution_id="bix-trace",
            trace_writer=trace_writer,
            max_iterations=2,
        )
    )

    assert result.status == "completed"
    events = (trace_writer.root_dir / "events.jsonl").read_text(encoding="utf-8")
    assert "agent_iteration_started" in events
    assert "agent_decision" in events
    assert "agent_execution_result" in events
    assert "agent_terminated" in events
