"""Deterministic aggregate execution implementation."""

import math
from typing import Final, Literal

from science_bot.pipeline.contracts import AggregateOperation
from science_bot.pipeline.execution.schemas import (
    AggregateExecutionInput,
    ExecutionStageOutput,
)
from science_bot.pipeline.execution.utils import (
    apply_resolved_filters,
    format_scalar_answer,
)

IMPLEMENTED_AGGREGATE_OPERATIONS: Final[frozenset[AggregateOperation]] = frozenset(
    {
        "count",
        "mean",
        "median",
        "variance",
        "skewness",
        "percentage",
        "proportion",
        "ratio",
    }
)


def run_aggregate_execution(payload: AggregateExecutionInput) -> ExecutionStageOutput:
    """Execute a resolved aggregate question.

    Args:
        payload: Resolved aggregate execution payload.

    Returns:
        ExecutionStageOutput: Deterministic aggregate result.
    """
    base_data = apply_resolved_filters(payload.data, payload.filters)

    if payload.operation == "count":
        count = int(len(base_data))
        return ExecutionStageOutput(
            family=payload.family,
            answer=str(count),
            raw_result={"count": count},
        )

    if payload.operation == "mean":
        value = float(base_data[payload.value_column].mean())
        return _numeric_output(
            payload.family,
            "mean",
            value,
            payload.decimal_places,
            payload.round_to,
        )

    if payload.operation == "median":
        value = float(base_data[payload.value_column].median())
        return _numeric_output(
            payload.family,
            "median",
            value,
            payload.decimal_places,
            payload.round_to,
        )

    if payload.operation == "variance":
        value = float(base_data[payload.value_column].var(ddof=1))
        return _numeric_output(
            payload.family,
            "variance",
            value,
            payload.decimal_places,
            payload.round_to,
        )

    if payload.operation == "skewness":
        value = float(base_data[payload.value_column].skew())
        return _numeric_output(
            payload.family,
            "skewness",
            value,
            payload.decimal_places,
            payload.round_to,
        )

    numerator, denominator = _count_fraction_parts(payload, base_data)
    if payload.operation in {"percentage", "proportion"}:
        value = numerator / denominator if denominator else math.nan
        raw_key = payload.operation
        answer = format_scalar_answer(
            value * 100 if payload.operation == "percentage" else value,
            payload.decimal_places,
            payload.round_to,
        )
        return ExecutionStageOutput(
            family=payload.family,
            answer=answer,
            raw_result={
                raw_key: value * 100 if payload.operation == "percentage" else value,
                "numerator": numerator,
                "denominator": denominator,
            },
        )

    ratio = numerator / denominator if denominator else math.nan
    return ExecutionStageOutput(
        family=payload.family,
        answer=format_scalar_answer(ratio, payload.decimal_places, payload.round_to),
        raw_result={
            "ratio": ratio,
            "numerator": numerator,
            "denominator": denominator,
        },
    )


def _count_fraction_parts(
    payload: AggregateExecutionInput,
    base_data,
) -> tuple[int, int]:
    """Count numerator and denominator rows for fraction-like aggregates.

    Args:
        payload: Aggregate execution payload.
        base_data: Dataframe after base filters have been applied.

    Returns:
        tuple[int, int]: Counted numerator and denominator values.
    """
    if payload.numerator_mask_column is not None:
        numerator = int(base_data[payload.numerator_mask_column].astype(bool).sum())
    else:
        numerator = int(
            len(apply_resolved_filters(base_data, payload.numerator_filters))
        )

    if payload.operation in {"percentage", "proportion"}:
        return numerator, int(len(base_data))

    if payload.denominator_mask_column is not None:
        denominator = int(base_data[payload.denominator_mask_column].astype(bool).sum())
    else:
        denominator = int(
            len(apply_resolved_filters(base_data, payload.denominator_filters))
        )
    return numerator, denominator


def _numeric_output(
    family: Literal["aggregate"],
    key: str,
    value: float,
    decimal_places: int | None,
    round_to: int | None,
) -> ExecutionStageOutput:
    """Build a numeric execution output.

    Args:
        family: Execution family name.
        key: Raw result field name.
        value: Numeric result.
        decimal_places: Optional rounding precision.
        round_to: Optional nearest-unit rounding.

    Returns:
        ExecutionStageOutput: Formatted output.
    """
    return ExecutionStageOutput(
        family=family,
        answer=format_scalar_answer(value, decimal_places, round_to),
        raw_result={key: value},
    )
