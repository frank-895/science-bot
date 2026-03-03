"""Execution stage router."""

from science_bot.pipeline.execution.aggregate import run_aggregate_execution
from science_bot.pipeline.execution.differential_expression import (
    run_differential_expression_execution,
)
from science_bot.pipeline.execution.hypothesis_test import (
    run_hypothesis_test_execution,
)
from science_bot.pipeline.execution.regression import run_regression_execution
from science_bot.pipeline.execution.schemas import (
    AggregateExecutionInput,
    DifferentialExpressionExecutionInput,
    ExecutionStageInput,
    ExecutionStageOutput,
    HypothesisTestExecutionInput,
    RegressionExecutionInput,
    VariantFilteringExecutionInput,
)
from science_bot.pipeline.execution.variant_filtering import (
    run_variant_filtering_execution,
)


def run_execution_stage(stage_input: ExecutionStageInput) -> ExecutionStageOutput:
    """Run the execution stage for a resolved family payload.

    Args:
        stage_input: Execution stage input.

    Returns:
        ExecutionStageOutput: Deterministic execution result.

    Raises:
        ValueError: If the execution family is unknown.
    """
    payload = stage_input.payload

    if isinstance(payload, AggregateExecutionInput):
        return run_aggregate_execution(payload)
    if isinstance(payload, HypothesisTestExecutionInput):
        return run_hypothesis_test_execution(payload)
    if isinstance(payload, RegressionExecutionInput):
        return run_regression_execution(payload)
    if isinstance(payload, DifferentialExpressionExecutionInput):
        return run_differential_expression_execution(payload)
    if isinstance(payload, VariantFilteringExecutionInput):
        return run_variant_filtering_execution(payload)

    raise ValueError(f"Unsupported execution family: {payload.family}")
