import asyncio
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError
from science_bot.pipeline.classification import ClassificationStageOutput
from science_bot.pipeline.contracts import (
    SupportedQuestionClassification,
    UnsupportedQuestionClassification,
)
from science_bot.pipeline.execution import ExecutionStageOutput
from science_bot.pipeline.execution.schemas import AggregateExecutionInput
from science_bot.pipeline.orchestrator import (
    OrchestratorRequest,
    run_orchestrator,
)
from science_bot.pipeline.resolution import (
    ResolutionStageOutput,
    ResolutionStepSummary,
)


def test_orchestrator_runs_all_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capsule_path = tmp_path / "capsule"
    capsule_path.mkdir()

    async def fake_run_classification_stage(
        stage_input: object,
    ) -> ClassificationStageOutput:
        return ClassificationStageOutput(
            classification=SupportedQuestionClassification(family="aggregate")
        )

    async def fake_run_resolution_stage(stage_input: object) -> ResolutionStageOutput:
        return ResolutionStageOutput(
            payload=AggregateExecutionInput(
                family="aggregate",
                operation="count",
                data=pd.DataFrame({"value": [1, 2, 3]}),
                filters=[],
                return_format="number",
            ),
            iterations_used=3,
            selected_files=["clinical.csv"],
            notes=["Picked clinical.csv as the primary table."],
            steps=[
                ResolutionStepSummary(
                    step_index=1,
                    kind="discover",
                    message="Shortlisted 4 candidate files.",
                    selected_files=[],
                    resolved_field_keys=[],
                )
            ],
        )

    def fake_run_execution_stage(stage_input: object) -> ExecutionStageOutput:
        return ExecutionStageOutput(
            family="aggregate",
            answer="42",
            raw_result={"value": 42},
            notes=["Execution completed."],
        )

    monkeypatch.setattr(
        "science_bot.pipeline.orchestrator.run_classification_stage",
        fake_run_classification_stage,
    )
    monkeypatch.setattr(
        "science_bot.pipeline.orchestrator.run_resolution_stage",
        fake_run_resolution_stage,
    )
    monkeypatch.setattr(
        "science_bot.pipeline.orchestrator.run_execution_stage",
        fake_run_execution_stage,
    )

    result = asyncio.run(
        run_orchestrator(
            OrchestratorRequest(question="What is this?", capsule_path=capsule_path)
        )
    )

    assert result.question == "What is this?"
    assert result.capsule_path == capsule_path
    assert result.status == "completed"
    assert result.answer == "42"
    assert result.metadata["classification_family"] == "aggregate"
    assert result.metadata["resolution_iterations_used"] == 3
    assert result.metadata["resolution_selected_files"] == ["clinical.csv"]
    assert result.metadata["execution_family"] == "aggregate"
    assert result.metadata["execution_raw_result"] == {"value": 42}
    assert result.metadata["execution_notes"] == ["Execution completed."]


def test_orchestrator_request_rejects_empty_question(tmp_path: Path) -> None:
    capsule_path = tmp_path / "capsule"
    capsule_path.mkdir()

    with pytest.raises(ValidationError, match="question must be non-empty"):
        OrchestratorRequest(question="   ", capsule_path=capsule_path)


def test_orchestrator_raises_for_missing_capsule(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing-capsule"

    with pytest.raises(FileNotFoundError, match="Capsule path not found"):
        asyncio.run(
            run_orchestrator(
                OrchestratorRequest(question="What is this?", capsule_path=missing_path)
            )
        )


def test_orchestrator_raises_for_unsupported_question(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capsule_path = tmp_path / "capsule"
    capsule_path.mkdir()

    async def fake_run_classification_stage(
        stage_input: object,
    ) -> ClassificationStageOutput:
        return ClassificationStageOutput(
            classification=UnsupportedQuestionClassification(
                family="unsupported",
                reason="Question is outside the supported analysis families.",
            )
        )

    monkeypatch.setattr(
        "science_bot.pipeline.orchestrator.run_classification_stage",
        fake_run_classification_stage,
    )

    with pytest.raises(ValueError, match="Unsupported question"):
        asyncio.run(
            run_orchestrator(
                OrchestratorRequest(
                    question="Write me a poem.",
                    capsule_path=capsule_path,
                )
            )
        )
