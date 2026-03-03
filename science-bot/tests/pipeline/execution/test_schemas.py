import pandas as pd
import pytest
from pydantic import ValidationError
from science_bot.pipeline.execution.schemas import (
    AggregateExecutionInput,
    ExecutionStageInput,
    HypothesisTestExecutionInput,
    ResolvedFilter,
)


def test_valid_execution_stage_input_for_aggregate() -> None:
    frame = pd.DataFrame({"value": [1, 2, 3]})

    parsed = ExecutionStageInput(
        payload=AggregateExecutionInput(
            family="aggregate",
            operation="median",
            data=frame,
            value_column="value",
            return_format="number",
        )
    )

    assert parsed.payload.family == "aggregate"


def test_aggregate_non_count_without_value_column_fails() -> None:
    frame = pd.DataFrame({"value": [1, 2, 3]})

    with pytest.raises(ValidationError):
        ExecutionStageInput.model_validate(
            {
                "payload": {
                    "family": "aggregate",
                    "operation": "median",
                    "data": frame,
                    "return_format": "number",
                }
            }
        )


def test_aggregate_skewness_without_value_column_fails() -> None:
    frame = pd.DataFrame({"value": [1, 2, 3]})

    with pytest.raises(ValidationError):
        ExecutionStageInput.model_validate(
            {
                "payload": {
                    "family": "aggregate",
                    "operation": "skewness",
                    "data": frame,
                    "return_format": "number",
                }
            }
        )


def test_aggregate_percentage_without_value_column_is_allowed() -> None:
    frame = pd.DataFrame({"selected": [True, False, True]})

    parsed = ExecutionStageInput(
        payload=AggregateExecutionInput(
            family="aggregate",
            operation="percentage",
            data=frame,
            numerator_mask_column="selected",
            return_format="percentage",
        )
    )

    assert parsed.payload.family == "aggregate"


def test_hypothesis_correlation_without_second_value_column_fails() -> None:
    frame = pd.DataFrame({"x": [1, 2], "y": [2, 3]})

    with pytest.raises(ValidationError):
        ExecutionStageInput.model_validate(
            {
                "payload": {
                    "family": "hypothesis_test",
                    "test": "pearson_correlation",
                    "data": frame,
                    "value_column": "x",
                    "return_field": "correlation",
                }
            }
        )


def test_cohens_d_without_group_values_fails() -> None:
    frame = pd.DataFrame({"group": ["a", "b"], "value": [1.0, 2.0]})

    with pytest.raises(ValidationError):
        ExecutionStageInput.model_validate(
            {
                "payload": {
                    "family": "hypothesis_test",
                    "test": "cohens_d",
                    "data": frame,
                    "value_column": "value",
                    "group_column": "group",
                    "return_field": "effect_size",
                }
            }
        )


def test_shapiro_wilk_allows_group_column_when_present() -> None:
    frame = pd.DataFrame({"group": ["KD", "KD"], "value": [1.0, 1.1]})

    parsed = ExecutionStageInput(
        payload=HypothesisTestExecutionInput(
            family="hypothesis_test",
            test="shapiro_wilk",
            data=frame,
            value_column="value",
            group_column="group",
            filters=[ResolvedFilter(column="group", operator="==", value="KD")],
            return_field="w_statistic",
        )
    )

    assert parsed.payload.family == "hypothesis_test"


def test_regression_polynomial_without_degree_fails() -> None:
    frame = pd.DataFrame({"x": [1, 2], "y": [2, 3]})

    with pytest.raises(ValidationError):
        ExecutionStageInput.model_validate(
            {
                "payload": {
                    "family": "regression",
                    "model_type": "polynomial",
                    "data": frame,
                    "outcome_column": "y",
                    "predictor_column": "x",
                    "return_field": "r_squared",
                }
            }
        )


def test_regression_non_polynomial_with_degree_fails() -> None:
    frame = pd.DataFrame({"x": [1, 2], "y": [2, 3]})

    with pytest.raises(ValidationError):
        ExecutionStageInput.model_validate(
            {
                "payload": {
                    "family": "regression",
                    "model_type": "linear",
                    "data": frame,
                    "outcome_column": "y",
                    "predictor_column": "x",
                    "degree": 3,
                    "return_field": "coefficient",
                }
            }
        )


def test_regression_predicted_probability_without_prediction_inputs_fails() -> None:
    frame = pd.DataFrame({"x": [1, 2], "y": [0, 1]})

    with pytest.raises(ValidationError):
        ExecutionStageInput.model_validate(
            {
                "payload": {
                    "family": "regression",
                    "model_type": "logistic",
                    "data": frame,
                    "outcome_column": "y",
                    "predictor_column": "x",
                    "return_field": "predicted_probability",
                }
            }
        )


def test_differential_expression_gene_log2fc_without_target_gene_fails() -> None:
    table = pd.DataFrame({"gene": ["A"], "log2FoldChange": [1.0], "padj": [0.01]})

    with pytest.raises(ValidationError):
        ExecutionStageInput.model_validate(
            {
                "payload": {
                    "family": "differential_expression",
                    "mode": "precomputed_results",
                    "operation": "gene_log2_fold_change",
                    "result_tables": {"comp": table},
                    "comparison_labels": ["comp"],
                }
            }
        )


def test_differential_expression_overlap_with_too_few_comparisons_fails() -> None:
    table = pd.DataFrame({"gene": ["A"], "log2FoldChange": [1.0], "padj": [0.01]})

    with pytest.raises(ValidationError):
        ExecutionStageInput.model_validate(
            {
                "payload": {
                    "family": "differential_expression",
                    "mode": "precomputed_results",
                    "operation": "shared_overlap_pattern",
                    "result_tables": {"comp": table},
                    "comparison_labels": ["comp"],
                }
            }
        )


def test_variant_filtering_invalid_vaf_range_fails() -> None:
    table = pd.DataFrame({"VAF": [0.1, 0.2]})

    with pytest.raises(ValidationError):
        ExecutionStageInput.model_validate(
            {
                "payload": {
                    "family": "variant_filtering",
                    "operation": "filtered_variant_count",
                    "data": table,
                    "vaf_column": "VAF",
                    "vaf_min": 0.8,
                    "vaf_max": 0.2,
                    "return_format": "number",
                }
            }
        )
