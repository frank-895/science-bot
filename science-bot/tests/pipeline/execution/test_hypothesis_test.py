import pandas as pd
from science_bot.pipeline.execution.hypothesis_test import run_hypothesis_test_execution
from science_bot.pipeline.execution.schemas import (
    HypothesisTestExecutionInput,
    ResolvedFilter,
)


def test_mann_whitney_u_statistic() -> None:
    frame = pd.DataFrame(
        {"group": ["a", "a", "b", "b"], "value": [1.0, 2.0, 10.0, 11.0]}
    )

    result = run_hypothesis_test_execution(
        HypothesisTestExecutionInput(
            family="hypothesis_test",
            test="mann_whitney_u",
            data=frame,
            value_column="value",
            group_column="group",
            group_a_value="a",
            group_b_value="b",
            return_field="u_statistic",
        )
    )

    assert result.answer == "0"


def test_t_test_statistic() -> None:
    frame = pd.DataFrame(
        {"group": ["a", "a", "b", "b"], "value": [1.0, 2.0, 10.0, 11.0]}
    )

    result = run_hypothesis_test_execution(
        HypothesisTestExecutionInput(
            family="hypothesis_test",
            test="t_test",
            data=frame,
            value_column="value",
            group_column="group",
            group_a_value="a",
            group_b_value="b",
            return_field="statistic",
            decimal_places=2,
        )
    )

    assert result.answer.startswith("-")


def test_chi_square_statistic() -> None:
    frame = pd.DataFrame({"group": ["a", "a", "b", "b"], "value": ["x", "x", "y", "y"]})

    result = run_hypothesis_test_execution(
        HypothesisTestExecutionInput(
            family="hypothesis_test",
            test="chi_square",
            data=frame,
            value_column="value",
            group_column="group",
            return_field="statistic",
            decimal_places=2,
        )
    )

    assert result.answer == "1.00"


def test_shapiro_wilk_w() -> None:
    frame = pd.DataFrame({"value": [1.0, 1.1, 0.9, 1.2, 0.95]})

    result = run_hypothesis_test_execution(
        HypothesisTestExecutionInput(
            family="hypothesis_test",
            test="shapiro_wilk",
            data=frame,
            value_column="value",
            return_field="w_statistic",
            decimal_places=3,
        )
    )

    assert len(result.answer.split(".")) == 2


def test_shapiro_wilk_applies_filters_even_with_group_column_present() -> None:
    frame = pd.DataFrame(
        {
            "group": ["KD", "KD", "KD", "control", "control"],
            "value": [1.0, 1.1, 0.9, 50.0, 60.0],
        }
    )

    result = run_hypothesis_test_execution(
        HypothesisTestExecutionInput(
            family="hypothesis_test",
            test="shapiro_wilk",
            data=frame,
            value_column="value",
            group_column="group",
            filters=[ResolvedFilter(column="group", operator="==", value="KD")],
            return_field="w_statistic",
            decimal_places=3,
        )
    )

    assert result.answer != "nan"


def test_pearson_correlation_coefficient() -> None:
    frame = pd.DataFrame({"x": [1, 2, 3, 4], "y": [2, 4, 6, 8]})

    result = run_hypothesis_test_execution(
        HypothesisTestExecutionInput(
            family="hypothesis_test",
            test="pearson_correlation",
            data=frame,
            value_column="x",
            second_value_column="y",
            return_field="correlation",
            decimal_places=2,
        )
    )

    assert result.answer == "1.00"


def test_cohens_d_effect_size() -> None:
    frame = pd.DataFrame(
        {"group": ["a", "a", "b", "b"], "value": [1.0, 2.0, 10.0, 11.0]}
    )

    result = run_hypothesis_test_execution(
        HypothesisTestExecutionInput(
            family="hypothesis_test",
            test="cohens_d",
            data=frame,
            value_column="value",
            group_column="group",
            group_a_value="a",
            group_b_value="b",
            return_field="effect_size",
            decimal_places=2,
        )
    )

    assert "effect_size" in result.raw_result
    assert result.answer.startswith("-")
