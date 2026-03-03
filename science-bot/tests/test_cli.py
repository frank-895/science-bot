import asyncio
from pathlib import Path

import pytest
from science_bot import cli
from science_bot.cli import (
    BenchmarkRow,
    BenchmarkSummary,
    load_benchmark_rows,
    resolve_benchmark_capsule_path,
    run_benchmark,
    score_benchmark_response,
)
from science_bot.pipeline.orchestrator import OrchestratorResult


def write_benchmark_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text(
        "\n".join(
            [
                "question,data_folder,capsule_uuid,question_id,ideal,eval_mode",
                *[
                    ",".join(
                        [
                            row["question"],
                            row["data_folder"],
                            row["capsule_uuid"],
                            row["question_id"],
                            row["ideal"],
                            row["eval_mode"],
                        ]
                    )
                    for row in rows
                ],
            ]
        ),
        encoding="utf-8",
    )


def test_build_parser_accepts_run_and_benchmark() -> None:
    parser = cli.build_parser()

    run_args = parser.parse_args(["run", "--question", "What?", "--capsule", "/tmp/x"])
    benchmark_args = parser.parse_args(["benchmark"])

    assert run_args.command == "run"
    assert run_args.question == "What?"
    assert run_args.capsule == "/tmp/x"
    assert benchmark_args.command == "benchmark"


def test_main_run_prints_human_readable_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    capsule_path = tmp_path / "capsule"
    capsule_path.mkdir()

    async def fake_run_orchestrator(request: object) -> OrchestratorResult:
        return OrchestratorResult(
            question="What?",
            capsule_path=capsule_path,
            status="completed",
            answer="stub-answer",
            metadata={"orchestrator_mode": "stub"},
            error=None,
        )

    monkeypatch.setattr(cli, "run_orchestrator", fake_run_orchestrator)

    exit_code = cli.main(["run", "--question", "What?", "--capsule", str(capsule_path)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Question: What?" in output
    assert f"Capsule: {capsule_path}" in output
    assert "Status: completed" in output
    assert "Answer: stub-answer" in output


def test_main_run_returns_error_for_missing_capsule(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = cli.main(
        ["run", "--question", "What?", "--capsule", "/definitely/missing/capsule"]
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "Error:" in output


def test_load_benchmark_rows_rejects_missing_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "benchmark.csv"
    csv_path.write_text("question,data_folder\nq1,folder\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        load_benchmark_rows(csv_path)


def test_resolve_benchmark_capsule_path_prefers_expected_match(tmp_path: Path) -> None:
    root = tmp_path / "extracted"
    expected = root / "cap-uuid" / "CapsuleData-inner-uuid"
    expected.mkdir(parents=True)

    row = BenchmarkRow(
        question="What?",
        data_folder="CapsuleFolder-inner-uuid.zip",
        capsule_uuid="cap-uuid",
        question_id="q1",
        ideal="answer",
        eval_mode="str_verifier",
    )

    resolved = resolve_benchmark_capsule_path(row, root)

    assert resolved == expected


def test_resolve_benchmark_capsule_path_uses_single_fallback(tmp_path: Path) -> None:
    root = tmp_path / "extracted"
    fallback = root / "cap-uuid" / "CapsuleData-other"
    fallback.mkdir(parents=True)

    row = BenchmarkRow(
        question="What?",
        data_folder="CapsuleFolder-inner-uuid.zip",
        capsule_uuid="cap-uuid",
        question_id="q1",
        ideal="answer",
        eval_mode="str_verifier",
    )

    resolved = resolve_benchmark_capsule_path(row, root)

    assert resolved == fallback


def test_resolve_benchmark_capsule_path_rejects_ambiguous_fallback(
    tmp_path: Path,
) -> None:
    root = tmp_path / "extracted"
    first = root / "cap-uuid" / "CapsuleData-a"
    second = root / "cap-uuid" / "CapsuleData-b"
    first.mkdir(parents=True)
    second.mkdir(parents=True)

    row = BenchmarkRow(
        question="What?",
        data_folder="CapsuleFolder-inner-uuid.zip",
        capsule_uuid="cap-uuid",
        question_id="q1",
        ideal="answer",
        eval_mode="str_verifier",
    )

    with pytest.raises(ValueError, match="Multiple CapsuleData"):
        resolve_benchmark_capsule_path(row, root)


def test_score_benchmark_response_for_supported_modes() -> None:
    assert score_benchmark_response("str_verifier", "  Hello  ", "hello")
    assert score_benchmark_response("range_verifier", "(1.5,1.7)", "value 1.6")
    assert not score_benchmark_response("range_verifier", "(1.5,1.7)", "value 2.1")
    assert score_benchmark_response("llm_verifier", "35%", "The result is 35%")


def test_run_benchmark_continues_after_row_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "benchmark.csv"
    extracted_root = tmp_path / "extracted"
    success_path = extracted_root / "cap-1" / "CapsuleData-inner-1"
    success_path.mkdir(parents=True)
    (extracted_root / "cap-2").mkdir(parents=True)
    write_benchmark_csv(
        csv_path,
        [
            {
                "question": "What is one?",
                "data_folder": "CapsuleFolder-inner-1.zip",
                "capsule_uuid": "cap-1",
                "question_id": "q1",
                "ideal": "ORCHESTRATOR_STUB_RESPONSE",
                "eval_mode": "str_verifier",
            },
            {
                "question": "What is two?",
                "data_folder": "CapsuleFolder-inner-2.zip",
                "capsule_uuid": "cap-2",
                "question_id": "q2",
                "ideal": "ORCHESTRATOR_STUB_RESPONSE",
                "eval_mode": "str_verifier",
            },
        ],
    )

    async def fake_run_orchestrator(request: object) -> OrchestratorResult:
        return OrchestratorResult(
            question="What is one?",
            capsule_path=success_path,
            status="completed",
            answer="ORCHESTRATOR_STUB_RESPONSE",
            metadata={"orchestrator_mode": "stub"},
            error=None,
        )

    monkeypatch.setattr(cli, "run_orchestrator", fake_run_orchestrator)

    summary = asyncio.run(run_benchmark(csv_path, extracted_root))

    assert isinstance(summary, BenchmarkSummary)
    assert summary.total_rows == 2
    assert summary.completed_rows == 1
    assert summary.failed_rows == 1
    assert summary.correct_rows == 1
    assert summary.incorrect_rows == 1
    assert any(
        row.question_id == "q2" and row.status == "failed" for row in summary.rows
    )


def test_main_benchmark_prints_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    summary = BenchmarkSummary(
        total_rows=2,
        completed_rows=2,
        failed_rows=0,
        correct_rows=1,
        incorrect_rows=1,
        accuracy=0.5,
        elapsed_seconds=0.123,
        rows=[],
    )

    async def fake_run_benchmark() -> BenchmarkSummary:
        return summary

    monkeypatch.setattr(cli, "run_benchmark", fake_run_benchmark)

    exit_code = cli.main(["benchmark"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Benchmark Summary" in output
    assert "Total rows: 2" in output
    assert "Accuracy: 50.00%" in output
