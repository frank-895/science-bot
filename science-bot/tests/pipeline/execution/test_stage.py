from typing import get_args

import pandas as pd
import science_bot.pipeline.execution.stage as execution_stage
from science_bot.pipeline.contracts import QuestionFamily
from science_bot.pipeline.execution.schemas import (
    AggregateExecutionInput,
    DifferentialExpressionExecutionInput,
    ExecutionStageInput,
    ExecutionStageOutput,
    HypothesisTestExecutionInput,
    RegressionExecutionInput,
    VariantFilteringExecutionInput,
)


def test_aggregate_payload_dispatches(monkeypatch) -> None:
    called = {}

    def fake_handler(payload):
        called["family"] = payload.family
        return ExecutionStageOutput(family="aggregate", answer="1")

    monkeypatch.setattr(execution_stage, "run_aggregate_execution", fake_handler)
    frame = pd.DataFrame({"value": [1]})

    result = execution_stage.run_execution_stage(
        ExecutionStageInput(
            payload=AggregateExecutionInput(
                family="aggregate",
                operation="count",
                data=frame,
                return_format="number",
            )
        )
    )

    assert called["family"] == "aggregate"
    assert result.answer == "1"


def test_hypothesis_payload_dispatches(monkeypatch) -> None:
    called = {}

    def fake_handler(payload):
        called["family"] = payload.family
        return ExecutionStageOutput(family="hypothesis_test", answer="2")

    monkeypatch.setattr(execution_stage, "run_hypothesis_test_execution", fake_handler)
    frame = pd.DataFrame({"group": ["a", "b"], "value": [1, 2]})

    result = execution_stage.run_execution_stage(
        ExecutionStageInput(
            payload=HypothesisTestExecutionInput(
                family="hypothesis_test",
                test="chi_square",
                data=frame,
                value_column="value",
                group_column="group",
                return_field="statistic",
            )
        )
    )

    assert called["family"] == "hypothesis_test"
    assert result.answer == "2"


def test_regression_payload_dispatches(monkeypatch) -> None:
    called = {}

    def fake_handler(payload):
        called["family"] = payload.family
        return ExecutionStageOutput(family="regression", answer="3")

    monkeypatch.setattr(execution_stage, "run_regression_execution", fake_handler)
    frame = pd.DataFrame({"x": [1, 2], "y": [2, 3]})

    result = execution_stage.run_execution_stage(
        ExecutionStageInput(
            payload=RegressionExecutionInput(
                family="regression",
                model_type="linear",
                data=frame,
                outcome_column="y",
                predictor_column="x",
                return_field="coefficient",
            )
        )
    )

    assert called["family"] == "regression"
    assert result.answer == "3"


def test_differential_expression_payload_dispatches(monkeypatch) -> None:
    called = {}

    def fake_handler(payload):
        called["family"] = payload.family
        return ExecutionStageOutput(family="differential_expression", answer="4")

    monkeypatch.setattr(
        execution_stage, "run_differential_expression_execution", fake_handler
    )
    table = pd.DataFrame({"gene": ["A"], "log2FoldChange": [1.0], "padj": [0.01]})

    result = execution_stage.run_execution_stage(
        ExecutionStageInput(
            payload=DifferentialExpressionExecutionInput(
                family="differential_expression",
                mode="precomputed_results",
                operation="significant_gene_count",
                result_tables={"comp": table},
                comparison_labels=["comp"],
            )
        )
    )

    assert called["family"] == "differential_expression"
    assert result.answer == "4"


def test_variant_filtering_payload_dispatches(monkeypatch) -> None:
    called = {}

    def fake_handler(payload):
        called["family"] = payload.family
        return ExecutionStageOutput(family="variant_filtering", answer="5")

    monkeypatch.setattr(
        execution_stage,
        "run_variant_filtering_execution",
        fake_handler,
    )
    frame = pd.DataFrame({"gene": ["A"], "VAF": [0.2]})

    result = execution_stage.run_execution_stage(
        ExecutionStageInput(
            payload=VariantFilteringExecutionInput(
                family="variant_filtering",
                operation="gene_with_max_variants",
                data=frame,
                gene_column="gene",
                return_format="label",
            )
        )
    )

    assert called["family"] == "variant_filtering"
    assert result.answer == "5"


def test_all_question_families_have_execution_implementations() -> None:
    families = set(get_args(QuestionFamily))
    implemented_families = {
        family
        for family in families
        if hasattr(execution_stage, f"run_{family}_execution")
    }

    assert implemented_families == families
