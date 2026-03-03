"""Deterministic regression execution implementation."""

from typing import Any, Final

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel

from science_bot.pipeline.contracts import (
    RegressionModelType,
    RegressionReturnField,
)
from science_bot.pipeline.execution.schemas import (
    ExecutionStageOutput,
    RegressionExecutionInput,
)
from science_bot.pipeline.execution.utils import (
    apply_resolved_filters,
    format_scalar_answer,
)

IMPLEMENTED_REGRESSION_MODEL_TYPES: Final[frozenset[RegressionModelType]] = frozenset(
    {
        "logistic",
        "ordinal_logistic",
        "linear",
        "polynomial",
    }
)
IMPLEMENTED_REGRESSION_RETURN_FIELDS: Final[frozenset[RegressionReturnField]] = (
    frozenset(
        {
            "odds_ratio",
            "percent_increase_in_odds",
            "predicted_probability",
            "coefficient",
            "r_squared",
        }
    )
)


def run_regression_execution(payload: RegressionExecutionInput) -> ExecutionStageOutput:
    """Execute a resolved regression question.

    Args:
        payload: Resolved regression execution payload.

    Returns:
        ExecutionStageOutput: Deterministic regression result.
    """
    data = apply_resolved_filters(payload.data, payload.filters)

    if payload.model_type == "linear":
        return _run_linear(payload, data)
    if payload.model_type == "polynomial":
        return _run_polynomial(payload, data)
    if payload.model_type == "logistic":
        return _run_logistic(payload, data)
    if payload.model_type == "ordinal_logistic":
        return _run_ordinal_logistic(payload, data)

    raise NotImplementedError(
        f"Regression model type '{payload.model_type}' is not implemented."
    )


def _run_linear(
    payload: RegressionExecutionInput, data: pd.DataFrame
) -> ExecutionStageOutput:
    """Run a linear regression model.

    Args:
        payload: Regression execution payload.
        data: Filtered dataframe.

    Returns:
        ExecutionStageOutput: Formatted regression output.
    """
    x, y = _prepare_numeric_regression_frame(payload, data)
    x = sm.add_constant(x)
    result = sm.OLS(y, x).fit()

    if payload.return_field == "coefficient":
        value = float(result.params[payload.predictor_column])
        raw_result = {"coefficient": value, "r_squared": float(result.rsquared)}
    else:
        value = float(result.rsquared)
        raw_result = {"r_squared": value}

    return ExecutionStageOutput(
        family=payload.family,
        answer=format_scalar_answer(value, payload.decimal_places, payload.round_to),
        raw_result=raw_result,
    )


def _run_polynomial(
    payload: RegressionExecutionInput, data: pd.DataFrame
) -> ExecutionStageOutput:
    """Run a polynomial regression model.

    Args:
        payload: Regression execution payload.
        data: Filtered dataframe.

    Returns:
        ExecutionStageOutput: Formatted regression output.
    """
    assert payload.degree is not None
    numeric_data, y = _prepare_numeric_regression_frame(payload, data)
    design = pd.DataFrame(index=numeric_data.index)
    for power in range(1, payload.degree + 1):
        design[f"{payload.predictor_column}^{power}"] = (
            numeric_data[payload.predictor_column] ** power
        )
    for column in payload.covariate_columns:
        design[column] = numeric_data[column]

    x = sm.add_constant(design)
    result = sm.OLS(y, x).fit()
    value = float(result.rsquared)
    return ExecutionStageOutput(
        family=payload.family,
        answer=format_scalar_answer(value, payload.decimal_places, payload.round_to),
        raw_result={"r_squared": value},
    )


def _run_logistic(
    payload: RegressionExecutionInput, data: pd.DataFrame
) -> ExecutionStageOutput:
    """Run a logistic regression model.

    Args:
        payload: Regression execution payload.
        data: Filtered dataframe.

    Returns:
        ExecutionStageOutput: Formatted regression output.
    """
    x, y = _prepare_numeric_regression_frame(payload, data)
    x = sm.add_constant(x)
    result = sm.Logit(y, x).fit(disp=False)
    return _build_logistic_output(payload, result)


def _run_ordinal_logistic(
    payload: RegressionExecutionInput, data: pd.DataFrame
) -> ExecutionStageOutput:
    """Run an ordinal logistic regression model.

    Args:
        payload: Regression execution payload.
        data: Filtered dataframe.

    Returns:
        ExecutionStageOutput: Formatted regression output.
    """
    x, y = _prepare_numeric_regression_frame(payload, data)
    result = OrderedModel(y, x, distr="logit").fit(method="bfgs", disp=False)
    return _build_logistic_output(payload, result)


def _build_logistic_output(
    payload: RegressionExecutionInput, result: Any
) -> ExecutionStageOutput:
    """Build output for logistic-style models.

    Args:
        payload: Regression execution payload.
        result: Fitted statsmodels result object.

    Returns:
        ExecutionStageOutput: Formatted execution output.
    """
    coefficient = float(result.params[payload.predictor_column])
    odds_ratio = float(np.exp(coefficient))

    if payload.return_field == "odds_ratio":
        value = odds_ratio
        raw_result = {"odds_ratio": odds_ratio, "coefficient": coefficient}
    elif payload.return_field == "percent_increase_in_odds":
        value = (odds_ratio - 1.0) * 100.0
        raw_result = {
            "percent_increase_in_odds": value,
            "odds_ratio": odds_ratio,
            "coefficient": coefficient,
        }
    elif payload.return_field == "predicted_probability":
        predict_frame = pd.DataFrame([_coerce_prediction_inputs(payload)])
        if "const" not in predict_frame and payload.model_type == "logistic":
            predict_frame = sm.add_constant(predict_frame, has_constant="add")
        value = float(result.predict(predict_frame)[0])
        raw_result = {"predicted_probability": value}
    else:
        value = coefficient
        raw_result = {"coefficient": coefficient, "odds_ratio": odds_ratio}

    return ExecutionStageOutput(
        family=payload.family,
        answer=format_scalar_answer(value, payload.decimal_places, payload.round_to),
        raw_result=raw_result,
    )


def _prepare_numeric_regression_frame(
    payload: RegressionExecutionInput,
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare numeric predictor and outcome data for regression execution.

    Args:
        payload: Regression execution payload.
        data: Filtered dataframe.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Numeric predictors and numeric outcome series.

    Raises:
        ValueError: If required columns cannot be coerced to usable numeric values.
    """
    required_columns = [
        payload.predictor_column,
        *payload.covariate_columns,
        payload.outcome_column,
    ]
    numeric_frame = (
        data[required_columns].apply(pd.to_numeric, errors="coerce").dropna()
    )
    if numeric_frame.empty:
        raise ValueError(
            "Regression requires numeric predictor/covariate/outcome columns after "
            "coercion."
        )
    if len(numeric_frame) < 2:
        raise ValueError(
            "Regression has insufficient numeric rows after filtering and coercion."
        )
    if payload.model_type in {"logistic", "ordinal_logistic"} and (
        numeric_frame[payload.outcome_column].nunique() < 2
    ):
        raise ValueError(
            "Regression requires at least two distinct outcome values after coercion."
        )
    predictors = numeric_frame[[payload.predictor_column, *payload.covariate_columns]]
    outcome = numeric_frame[payload.outcome_column]
    return predictors, outcome


def _coerce_prediction_inputs(
    payload: RegressionExecutionInput,
) -> dict[str, str | int | float | bool]:
    """Coerce numeric prediction inputs for logistic prediction.

    Args:
        payload: Regression execution payload.

    Returns:
        dict[str, str | int | float | bool]: Prediction inputs with numeric values
        coerced when required.

    Raises:
        ValueError: If a required numeric prediction field cannot be coerced.
    """
    coerced: dict[str, str | int | float | bool] = {}
    numeric_fields = {payload.predictor_column, *payload.covariate_columns}
    for field, value in payload.prediction_inputs.items():
        if field not in numeric_fields or isinstance(value, bool):
            coerced[field] = value
            continue
        numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric_value):
            raise ValueError(
                "predicted_probability requires numeric prediction input for field "
                f"'{field}'."
            )
        coerced[field] = float(numeric_value)
    return coerced
