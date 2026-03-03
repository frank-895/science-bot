import pandas as pd
import pytest
from science_bot.pipeline.execution.aggregate import run_aggregate_execution
from science_bot.pipeline.execution.schemas import (
    AggregateExecutionInput,
    ResolvedFilter,
)


def test_count_after_filters() -> None:
    frame = pd.DataFrame({"group": ["a", "b", "a"]})

    result = run_aggregate_execution(
        AggregateExecutionInput(
            family="aggregate",
            operation="count",
            data=frame,
            filters=[ResolvedFilter(column="group", operator="==", value="a")],
            return_format="number",
        )
    )

    assert result.answer == "2"


def test_median_numeric_column() -> None:
    frame = pd.DataFrame({"value": [1.0, 2.0, 9.0]})

    result = run_aggregate_execution(
        AggregateExecutionInput(
            family="aggregate",
            operation="median",
            data=frame,
            value_column="value",
            return_format="number",
        )
    )

    assert result.answer == "2"


def test_mean_numeric_column() -> None:
    frame = pd.DataFrame({"value": [1.0, 2.0, 3.0]})

    result = run_aggregate_execution(
        AggregateExecutionInput(
            family="aggregate",
            operation="mean",
            data=frame,
            value_column="value",
            return_format="number",
            decimal_places=2,
        )
    )

    assert result.answer == "2.00"


def test_mean_rounds_to_nearest_thousand() -> None:
    frame = pd.DataFrame({"value": [82442.3333333333]})

    result = run_aggregate_execution(
        AggregateExecutionInput(
            family="aggregate",
            operation="mean",
            data=frame,
            value_column="value",
            return_format="number",
            round_to=1000,
        )
    )

    assert result.answer == "82000"


def test_skewness_numeric_column() -> None:
    frame = pd.DataFrame({"value": [1.0, 2.0, 9.0]})

    result = run_aggregate_execution(
        AggregateExecutionInput(
            family="aggregate",
            operation="skewness",
            data=frame,
            value_column="value",
            return_format="number",
            decimal_places=3,
        )
    )

    assert "skewness" in result.raw_result
    assert "." in result.answer


def test_percentage_calculation() -> None:
    frame = pd.DataFrame({"selected": [True, False, True, False]})

    result = run_aggregate_execution(
        AggregateExecutionInput(
            family="aggregate",
            operation="percentage",
            data=frame,
            value_column="selected",
            numerator_mask_column="selected",
            return_format="percentage",
        )
    )

    assert result.answer == "50"


def test_percentage_calculation_without_value_column() -> None:
    frame = pd.DataFrame({"selected": [True, False, True, False]})

    result = run_aggregate_execution(
        AggregateExecutionInput(
            family="aggregate",
            operation="percentage",
            data=frame,
            numerator_mask_column="selected",
            return_format="percentage",
        )
    )

    assert result.answer == "50"


def test_ratio_calculation() -> None:
    frame = pd.DataFrame(
        {"numerator": [True, True, False], "denominator": [True, True, True]}
    )

    result = run_aggregate_execution(
        AggregateExecutionInput(
            family="aggregate",
            operation="ratio",
            data=frame,
            value_column="numerator",
            numerator_mask_column="numerator",
            denominator_mask_column="denominator",
            return_format="ratio",
            decimal_places=2,
        )
    )

    assert result.answer == "0.67"


def test_percentage_calculation_from_filters() -> None:
    frame = pd.DataFrame({"group": ["a", "a", "b", "b"]})

    result = run_aggregate_execution(
        AggregateExecutionInput(
            family="aggregate",
            operation="percentage",
            data=frame,
            numerator_filters=[
                ResolvedFilter(column="group", operator="==", value="a")
            ],
            return_format="percentage",
        )
    )

    assert result.answer == "50"


def test_ratio_calculation_from_filters() -> None:
    frame = pd.DataFrame({"group": ["a", "a", "b", "b", "b"]})

    result = run_aggregate_execution(
        AggregateExecutionInput(
            family="aggregate",
            operation="ratio",
            data=frame,
            numerator_filters=[
                ResolvedFilter(column="group", operator="==", value="a")
            ],
            denominator_filters=[
                ResolvedFilter(column="group", operator="==", value="b")
            ],
            return_format="ratio",
            decimal_places=2,
        )
    )

    assert result.answer == "0.67"


def test_ratio_requires_denominator_mask_or_filters() -> None:
    frame = pd.DataFrame({"group": ["a", "b"]})

    with pytest.raises(
        ValueError,
        match="ratio requires denominator_mask_column or denominator_filters.",
    ):
        AggregateExecutionInput(
            family="aggregate",
            operation="ratio",
            data=frame,
            numerator_filters=[
                ResolvedFilter(column="group", operator="==", value="a")
            ],
            return_format="ratio",
        )
