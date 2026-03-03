import pandas as pd
import pytest
from science_bot.pipeline.execution.differential_expression import (
    run_differential_expression_execution,
)
from science_bot.pipeline.execution.schemas import DifferentialExpressionExecutionInput


def test_significant_gene_count() -> None:
    table = pd.DataFrame(
        {
            "gene": ["a", "b", "c"],
            "log2FoldChange": [2.0, 0.2, -3.0],
            "padj": [0.01, 0.01, 0.2],
            "baseMean": [11, 11, 11],
        }
    )

    result = run_differential_expression_execution(
        DifferentialExpressionExecutionInput(
            family="differential_expression",
            mode="precomputed_results",
            operation="significant_gene_count",
            result_tables={"comp": table},
            comparison_labels=["comp"],
            adjusted_p_value_column="padj",
            log_fold_change_column="log2FoldChange",
            base_mean_column="baseMean",
            significance_threshold=0.05,
            fold_change_threshold=1.0,
            basemean_threshold=10.0,
        )
    )

    assert result.answer == "1"


def test_unique_significant_gene_count() -> None:
    primary = pd.DataFrame(
        {
            "gene": ["a", "b"],
            "log2FoldChange": [2.0, 2.0],
            "padj": [0.01, 0.01],
            "baseMean": [11, 11],
        }
    )
    secondary = pd.DataFrame(
        {
            "gene": ["b", "c"],
            "log2FoldChange": [2.0, 2.0],
            "padj": [0.01, 0.01],
            "baseMean": [11, 11],
        }
    )

    result = run_differential_expression_execution(
        DifferentialExpressionExecutionInput(
            family="differential_expression",
            mode="precomputed_results",
            operation="unique_significant_gene_count",
            result_tables={"primary": primary, "secondary": secondary},
            comparison_labels=["primary", "secondary"],
            gene_column="gene",
            adjusted_p_value_column="padj",
            log_fold_change_column="log2FoldChange",
            base_mean_column="baseMean",
            significance_threshold=0.05,
            fold_change_threshold=1.0,
            basemean_threshold=10.0,
        )
    )

    assert result.answer == "1"


def test_shared_overlap_pattern_label() -> None:
    first = pd.DataFrame(
        {
            "gene": ["a"],
            "log2FoldChange": [2.0],
            "padj": [0.01],
            "baseMean": [11],
        }
    )
    second = pd.DataFrame(
        {
            "gene": ["b"],
            "log2FoldChange": [2.0],
            "padj": [0.01],
            "baseMean": [11],
        }
    )

    result = run_differential_expression_execution(
        DifferentialExpressionExecutionInput(
            family="differential_expression",
            mode="precomputed_results",
            operation="shared_overlap_pattern",
            result_tables={"first": first, "second": second},
            comparison_labels=["first", "second"],
            gene_column="gene",
            adjusted_p_value_column="padj",
            log_fold_change_column="log2FoldChange",
            base_mean_column="baseMean",
            significance_threshold=0.05,
            fold_change_threshold=1.0,
            basemean_threshold=10.0,
        )
    )

    assert result.answer == "No overlap between any groups"


def test_gene_log2_fold_change_lookup() -> None:
    table = pd.DataFrame(
        {
            "gene": ["PA14_35160"],
            "log2FoldChange": [-4.1],
            "padj": [0.01],
            "baseMean": [11],
        }
    )

    result = run_differential_expression_execution(
        DifferentialExpressionExecutionInput(
            family="differential_expression",
            mode="precomputed_results",
            operation="gene_log2_fold_change",
            result_tables={"comp": table},
            comparison_labels=["comp"],
            target_gene="PA14_35160",
            gene_column="gene",
            log_fold_change_column="log2FoldChange",
            decimal_places=2,
        )
    )

    assert result.answer == "-4.10"


def test_correction_ratio() -> None:
    bonferroni = pd.DataFrame({"adjusted_p_value": [0.2, 0.3]})
    by = pd.DataFrame({"adjusted_p_value": [0.2, 0.3]})

    result = run_differential_expression_execution(
        DifferentialExpressionExecutionInput(
            family="differential_expression",
            mode="precomputed_results",
            operation="correction_ratio",
            result_tables={"bonferroni": bonferroni, "by": by},
            comparison_labels=["markers"],
            correction_methods=["bonferroni", "by"],
            adjusted_p_value_column="adjusted_p_value",
            significance_threshold=0.05,
        )
    )

    assert result.answer == "0:0"


def test_raw_counts_mode_is_not_implemented() -> None:
    with pytest.raises(
        NotImplementedError,
        match="Differential expression execution mode 'raw_counts' is not implemented.",
    ):
        run_differential_expression_execution(
            DifferentialExpressionExecutionInput(
                family="differential_expression",
                mode="raw_counts",
                operation="significant_gene_count",
                count_matrix=pd.DataFrame({"sample": [1]}),
                sample_metadata=pd.DataFrame({"sample": [1]}),
                comparison_labels=["comp"],
            )
        )


def test_significant_gene_count_supports_noncanonical_columns() -> None:
    table = pd.DataFrame(
        {
            "feature_id": ["a", "b", "c"],
            "lfc": [2.0, 0.2, -3.0],
            "fdr": [0.01, 0.01, 0.2],
            "mean_count": [11, 11, 11],
        }
    )

    result = run_differential_expression_execution(
        DifferentialExpressionExecutionInput(
            family="differential_expression",
            mode="precomputed_results",
            operation="significant_gene_count",
            result_tables={"comp": table},
            comparison_labels=["comp"],
            adjusted_p_value_column="fdr",
            log_fold_change_column="lfc",
            base_mean_column="mean_count",
            significance_threshold=0.05,
            fold_change_threshold=1.0,
            basemean_threshold=10.0,
        )
    )

    assert result.answer == "1"


def test_gene_log2_fold_change_requires_resolved_columns() -> None:
    table = pd.DataFrame({"gene": ["PA14_35160"], "log2FoldChange": [-4.1]})

    with pytest.raises(
        ValueError,
        match="Differential expression execution requires gene_column.",
    ):
        run_differential_expression_execution(
            DifferentialExpressionExecutionInput(
                family="differential_expression",
                mode="precomputed_results",
                operation="gene_log2_fold_change",
                result_tables={"comp": table},
                comparison_labels=["comp"],
                target_gene="PA14_35160",
            )
        )
