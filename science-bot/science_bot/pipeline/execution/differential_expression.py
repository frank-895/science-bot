"""Deterministic differential expression execution implementation."""

from typing import Final

import pandas as pd

from science_bot.pipeline.contracts import DifferentialExpressionOperation
from science_bot.pipeline.execution.schemas import (
    DifferentialExpressionExecutionInput,
    DifferentialExpressionExecutionMode,
    ExecutionStageOutput,
)
from science_bot.pipeline.execution.utils import format_scalar_answer

IMPLEMENTED_DIFFERENTIAL_EXPRESSION_OPERATIONS: Final[
    frozenset[DifferentialExpressionOperation]
] = frozenset(
    {
        "significant_gene_count",
        "unique_significant_gene_count",
        "shared_overlap_pattern",
        "gene_log2_fold_change",
        "significant_marker_count",
        "correction_ratio",
    }
)
IMPLEMENTED_DIFFERENTIAL_EXPRESSION_EXECUTION_MODES: Final[
    frozenset[DifferentialExpressionExecutionMode]
] = frozenset({"precomputed_results"})


def run_differential_expression_execution(
    payload: DifferentialExpressionExecutionInput,
) -> ExecutionStageOutput:
    """Execute a resolved differential expression question.

    Args:
        payload: Resolved differential expression execution payload.

    Returns:
        ExecutionStageOutput: Deterministic differential expression result.

    Raises:
        NotImplementedError: If raw-count mode is requested.
        KeyError: If a required comparison table is missing.
    """
    if payload.mode == "raw_counts":
        raise NotImplementedError(
            "Differential expression execution mode 'raw_counts' is not implemented."
        )

    if payload.operation == "significant_gene_count":
        table = payload.result_tables[payload.comparison_labels[0]]
        filtered = _filter_significant(table, payload)
        count = int(len(filtered))
        return ExecutionStageOutput(
            family=payload.family,
            answer=str(count),
            raw_result={"significant_gene_count": count},
        )

    if payload.operation == "unique_significant_gene_count":
        primary = _significant_gene_set(
            payload.result_tables[payload.comparison_labels[0]], payload
        )
        others = set()
        for label in payload.comparison_labels[1:]:
            others |= _significant_gene_set(payload.result_tables[label], payload)
        count = int(len(primary - others))
        return ExecutionStageOutput(
            family=payload.family,
            answer=str(count),
            raw_result={"unique_significant_gene_count": count},
        )

    if payload.operation == "shared_overlap_pattern":
        sets = [
            _significant_gene_set(payload.result_tables[label], payload)
            for label in payload.comparison_labels
        ]
        shared_all = set.intersection(*sets)
        if shared_all:
            answer = "Complete overlap between all groups"
        elif _has_partial_overlap(sets):
            answer = "Partial overlap between specific groups"
        else:
            answer = "No overlap between any groups"
        return ExecutionStageOutput(
            family=payload.family,
            answer=answer,
            raw_result={"shared_overlap_pattern": answer},
        )

    if payload.operation == "gene_log2_fold_change":
        table = payload.result_tables[payload.comparison_labels[0]]
        gene_column = _require_gene_column(payload)
        log_fold_change_column = _require_log_fold_change_column(payload)
        row = table.loc[table[gene_column] == payload.target_gene].iloc[0]
        value = float(row[log_fold_change_column])
        return ExecutionStageOutput(
            family=payload.family,
            answer=format_scalar_answer(
                value, payload.decimal_places, payload.round_to
            ),
            raw_result={"gene_log2_fold_change": value},
        )

    if payload.operation == "significant_marker_count":
        total = 0
        for label in payload.comparison_labels:
            total += len(_filter_significant(payload.result_tables[label], payload))
        return ExecutionStageOutput(
            family=payload.family,
            answer=str(int(total)),
            raw_result={"significant_marker_count": int(total)},
        )

    counts: list[int] = []
    adjusted_p_value_column = _require_adjusted_p_value_column(payload)
    for method in payload.correction_methods:
        table = payload.result_tables[method]
        counts.append(
            int((table[adjusted_p_value_column] < payload.significance_threshold).sum())
        )
    answer = f"{counts[0]}:{counts[1]}"
    return ExecutionStageOutput(
        family=payload.family,
        answer=answer,
        raw_result={"correction_ratio": answer, "counts": counts},
    )

    raise NotImplementedError(
        f"Differential expression operation '{payload.operation}' is not implemented."
    )


def _filter_significant(
    table: pd.DataFrame, payload: DifferentialExpressionExecutionInput
) -> pd.DataFrame:
    """Apply significance thresholds to a DE results table.

    Args:
        table: Differential expression results table.
        payload: Differential expression execution payload.

    Returns:
        pd.DataFrame: Thresholded DE result table.
    """
    filtered = table.copy()
    if payload.significance_threshold is not None:
        adjusted_p_value_column = _require_adjusted_p_value_column(payload)
        filtered = filtered[
            filtered[adjusted_p_value_column] < payload.significance_threshold
        ]
    if payload.fold_change_threshold is not None:
        log_fold_change_column = _require_log_fold_change_column(payload)
        filtered = filtered[
            filtered[log_fold_change_column].abs() > payload.fold_change_threshold
        ]
    if payload.basemean_threshold is not None:
        base_mean_column = _require_base_mean_column(payload)
        filtered = filtered[filtered[base_mean_column] >= payload.basemean_threshold]
    return filtered


def _significant_gene_set(
    table: pd.DataFrame, payload: DifferentialExpressionExecutionInput
) -> set[str]:
    """Build the significant gene set for one comparison table.

    Args:
        table: Differential expression results table.
        payload: Differential expression execution payload.

    Returns:
        set[str]: Significant gene identifiers.
    """
    gene_column = _require_gene_column(payload)
    return set(_filter_significant(table, payload)[gene_column].astype(str))


def _require_gene_column(payload: DifferentialExpressionExecutionInput) -> str:
    """Return the resolved gene column for DE execution.

    Args:
        payload: Differential expression execution payload.

    Returns:
        str: Resolved gene column name.

    Raises:
        ValueError: If the gene column was not resolved.
    """
    if payload.gene_column is None:
        raise ValueError("Differential expression execution requires gene_column.")
    return payload.gene_column


def _require_log_fold_change_column(
    payload: DifferentialExpressionExecutionInput,
) -> str:
    """Return the resolved log fold change column for DE execution.

    Args:
        payload: Differential expression execution payload.

    Returns:
        str: Resolved log fold change column name.

    Raises:
        ValueError: If the log fold change column was not resolved.
    """
    if payload.log_fold_change_column is None:
        raise ValueError(
            "Differential expression execution requires log_fold_change_column."
        )
    return payload.log_fold_change_column


def _require_adjusted_p_value_column(
    payload: DifferentialExpressionExecutionInput,
) -> str:
    """Return the resolved adjusted p-value column for DE execution.

    Args:
        payload: Differential expression execution payload.

    Returns:
        str: Resolved adjusted p-value column name.

    Raises:
        ValueError: If the adjusted p-value column was not resolved.
    """
    if payload.adjusted_p_value_column is None:
        raise ValueError(
            "Differential expression execution requires adjusted_p_value_column."
        )
    return payload.adjusted_p_value_column


def _require_base_mean_column(payload: DifferentialExpressionExecutionInput) -> str:
    """Return the resolved base mean column for DE execution.

    Args:
        payload: Differential expression execution payload.

    Returns:
        str: Resolved base mean column name.

    Raises:
        ValueError: If the base mean column was not resolved.
    """
    if payload.base_mean_column is None:
        raise ValueError("Differential expression execution requires base_mean_column.")
    return payload.base_mean_column


def _has_partial_overlap(sets: list[set[str]]) -> bool:
    """Determine whether any pairwise overlap exists without full overlap.

    Args:
        sets: Significant gene sets per comparison.

    Returns:
        bool: True when a partial overlap exists.
    """
    for index, left in enumerate(sets):
        for right in sets[index + 1 :]:
            if left & right:
                return True
    return False
