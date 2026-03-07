import asyncio
from pathlib import Path

import pytest
from pydantic import ValidationError
from science_bot.agent.contracts import AgentDecision
from science_bot.agent.runtime import run_agent
from science_bot.providers.executor import PythonExecutionResult


def test_run_agent_records_python_then_respond_steps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    decisions = [
        AgentDecision(decision="run_python", script="print('x')"),
        AgentDecision(decision="respond", answer="42"),
        AgentDecision(decision="respond", answer="43"),
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
        del run_id
        assert "print" in script
        return PythonExecutionResult(
            status="completed",
            answer=None,
            error_type=None,
            error_message=None,
            stdout_tail="ok",
            stderr_tail="",
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
            max_iterations=3,
        )
    )

    assert result.status == "completed"
    assert result.answer == "43"
    assert result.iterations_used == 3
    assert len(result.steps) == 3
    assert result.steps[0].decision == "run_python"
    assert result.steps[0].execution_status == "completed"
    assert result.steps[2].decision == "respond"
    assert result.steps[2].answer == "43"


def test_run_agent_need_info_is_terminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def fake_parse_structured(**_: object) -> AgentDecision:
        return AgentDecision(decision="need_info", reason="Missing cohort labels")

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
        AgentDecision(decision="respond", answer="candidate-1"),
        AgentDecision(decision="run_python", script="print('hi')"),
        AgentDecision(decision="respond", answer="candidate-2"),
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
        return AgentDecision(decision="run_python", script="print(1)")

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
            max_iterations=2,
        )
    )

    assert result.status == "failed"
    assert result.failure_reason == "max_iterations_no_answer"
    assert result.iterations_used == 2
    assert len(result.steps) == 2


def test_agent_decision_rejects_invalid_field_combinations() -> None:
    with pytest.raises(ValidationError):
        AgentDecision(decision="run_python", script="print(1)", answer="not-allowed")
    with pytest.raises(ValidationError):
        AgentDecision(decision="respond", answer="x", reason="not-allowed")
    with pytest.raises(ValidationError):
        AgentDecision(decision="need_info", reason="x", script="print(1)")
