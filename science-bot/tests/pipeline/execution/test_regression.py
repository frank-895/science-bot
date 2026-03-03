import numpy as np
import pandas as pd
import pytest
from science_bot.pipeline.execution.regression import run_regression_execution
from science_bot.pipeline.execution.schemas import RegressionExecutionInput


def test_logistic_odds_ratio() -> None:
    frame = pd.DataFrame({"x": [0, 1, 1, 2, 2, 3, 3, 4], "y": [0, 0, 1, 0, 1, 1, 0, 1]})

    result = run_regression_execution(
        RegressionExecutionInput(
            family="regression",
            model_type="logistic",
            data=frame,
            outcome_column="y",
            predictor_column="x",
            return_field="odds_ratio",
            decimal_places=2,
        )
    )

    assert float(result.answer) > 1.0


def test_ordinal_logistic_odds_ratio() -> None:
    frame = pd.DataFrame(
        {
            "x": [0, 0, 1, 1, 2, 2, 3, 3],
            "y": [0, 0, 1, 1, 1, 2, 2, 2],
        }
    )

    result = run_regression_execution(
        RegressionExecutionInput(
            family="regression",
            model_type="ordinal_logistic",
            data=frame,
            outcome_column="y",
            predictor_column="x",
            return_field="odds_ratio",
            decimal_places=2,
        )
    )

    assert float(result.answer) > 1.0


def test_linear_coefficient() -> None:
    frame = pd.DataFrame({"x": [1, 2, 3, 4], "y": [2, 4, 6, 8]})

    result = run_regression_execution(
        RegressionExecutionInput(
            family="regression",
            model_type="linear",
            data=frame,
            outcome_column="y",
            predictor_column="x",
            return_field="coefficient",
            decimal_places=2,
        )
    )

    assert result.answer == "2.00"


def test_polynomial_r_squared() -> None:
    x = np.array([0, 1, 2, 3, 4], dtype=float)
    frame = pd.DataFrame({"x": x, "y": x**2 + 1})

    result = run_regression_execution(
        RegressionExecutionInput(
            family="regression",
            model_type="polynomial",
            data=frame,
            outcome_column="y",
            predictor_column="x",
            degree=2,
            return_field="r_squared",
            decimal_places=2,
        )
    )

    assert result.answer == "1.00"


def test_predicted_probability_formatting() -> None:
    frame = pd.DataFrame({"x": [0, 1, 1, 2, 2, 3, 3, 4], "y": [0, 0, 1, 0, 1, 1, 0, 1]})

    result = run_regression_execution(
        RegressionExecutionInput(
            family="regression",
            model_type="logistic",
            data=frame,
            outcome_column="y",
            predictor_column="x",
            prediction_inputs={"x": 1},
            return_field="predicted_probability",
            decimal_places=2,
        )
    )

    assert 0.0 <= float(result.answer) <= 1.0


def test_linear_regression_coerces_numeric_strings() -> None:
    frame = pd.DataFrame({"x": ["1", "2", "3", "4"], "y": ["2", "4", "6", "8"]})

    result = run_regression_execution(
        RegressionExecutionInput(
            family="regression",
            model_type="linear",
            data=frame,
            outcome_column="y",
            predictor_column="x",
            return_field="coefficient",
            decimal_places=2,
        )
    )

    assert result.answer == "2.00"


def test_polynomial_regression_rejects_nonnumeric_predictor() -> None:
    frame = pd.DataFrame({"x": ["a", "b", "c"], "y": [1, 2, 3]})

    with pytest.raises(
        ValueError,
        match=(
            "Regression requires numeric predictor/covariate/outcome "
            "columns after coercion."
        ),
    ):
        run_regression_execution(
            RegressionExecutionInput(
                family="regression",
                model_type="polynomial",
                data=frame,
                outcome_column="y",
                predictor_column="x",
                degree=2,
                return_field="r_squared",
            )
        )


def test_predicted_probability_rejects_nonnumeric_prediction_input() -> None:
    frame = pd.DataFrame({"x": [0, 1, 1, 2, 2, 3, 3, 4], "y": [0, 0, 1, 0, 1, 1, 0, 1]})

    with pytest.raises(
        ValueError,
        match="predicted_probability requires numeric prediction input for field 'x'.",
    ):
        run_regression_execution(
            RegressionExecutionInput(
                family="regression",
                model_type="logistic",
                data=frame,
                outcome_column="y",
                predictor_column="x",
                prediction_inputs={"x": "abc"},
                return_field="predicted_probability",
            )
        )
