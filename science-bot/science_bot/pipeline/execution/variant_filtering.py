"""Deterministic variant filtering execution implementation."""

import math
from typing import Final

import pandas as pd

from science_bot.pipeline.contracts import VariantFilteringOperation
from science_bot.pipeline.execution.schemas import (
    ExecutionStageOutput,
    VariantFilteringExecutionInput,
)
from science_bot.pipeline.execution.utils import (
    apply_resolved_filters,
    format_scalar_answer,
)

IMPLEMENTED_VARIANT_FILTERING_OPERATIONS: Final[
    frozenset[VariantFilteringOperation]
] = frozenset(
    {
        "filtered_variant_count",
        "variant_fraction",
        "variant_proportion",
        "gene_with_max_variants",
        "sample_variant_count",
    }
)


def run_variant_filtering_execution(
    payload: VariantFilteringExecutionInput,
) -> ExecutionStageOutput:
    """Execute a resolved variant filtering question.

    Args:
        payload: Resolved variant filtering execution payload.

    Returns:
        ExecutionStageOutput: Deterministic variant filtering result.
    """
    data = apply_resolved_filters(payload.data, payload.filters)
    data = _apply_variant_bounds(data, payload)

    if payload.operation == "filtered_variant_count":
        count = int(len(data))
        return ExecutionStageOutput(
            family=payload.family,
            answer=str(count),
            raw_result={"filtered_variant_count": count},
        )

    if payload.operation == "sample_variant_count":
        sample_data = data.loc[data[payload.sample_column] == payload.sample_value]
        count = int(len(sample_data))
        return ExecutionStageOutput(
            family=payload.family,
            answer=str(count),
            raw_result={"sample_variant_count": count},
        )

    if payload.operation == "gene_with_max_variants":
        top_gene = str(data[payload.gene_column].value_counts().idxmax())
        return ExecutionStageOutput(
            family=payload.family,
            answer=top_gene,
            raw_result={"gene_with_max_variants": top_gene},
        )

    numerator = int(data[payload.effect_column].astype(bool).sum())
    denominator = int(len(data))
    value = numerator / denominator if denominator else math.nan

    if payload.return_format == "percentage":
        display_value = value * 100.0
    else:
        display_value = value

    key = (
        "variant_fraction"
        if payload.operation == "variant_fraction"
        else "variant_proportion"
    )
    return ExecutionStageOutput(
        family=payload.family,
        answer=format_scalar_answer(
            display_value, payload.decimal_places, payload.round_to
        ),
        raw_result={
            key: display_value,
            "numerator": numerator,
            "denominator": denominator,
        },
    )


def _apply_variant_bounds(
    data: pd.DataFrame, payload: VariantFilteringExecutionInput
) -> pd.DataFrame:
    """Apply VAF range filters to a dataframe.

    Args:
        data: Filtered dataframe.
        payload: Variant filtering execution payload.

    Returns:
        pd.DataFrame: VAF-bounded dataframe.
    """
    bounded = data
    if payload.vaf_column is not None and (
        payload.vaf_min is not None or payload.vaf_max is not None
    ):
        bounded = bounded.copy()
        bounded[payload.vaf_column] = pd.to_numeric(
            bounded[payload.vaf_column], errors="coerce"
        )
    if payload.vaf_min is not None:
        bounded = bounded[bounded[payload.vaf_column] >= payload.vaf_min]
    if payload.vaf_max is not None:
        bounded = bounded[bounded[payload.vaf_column] <= payload.vaf_max]
    return bounded
