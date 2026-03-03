import pandas as pd
from science_bot.pipeline.execution.schemas import VariantFilteringExecutionInput
from science_bot.pipeline.execution.variant_filtering import (
    run_variant_filtering_execution,
)


def test_filtered_count() -> None:
    frame = pd.DataFrame({"VAF": [0.1, 0.2, 0.8]})

    result = run_variant_filtering_execution(
        VariantFilteringExecutionInput(
            family="variant_filtering",
            operation="filtered_variant_count",
            data=frame,
            vaf_column="VAF",
            vaf_max=0.3,
            return_format="number",
        )
    )

    assert result.answer == "2"


def test_fraction_after_vaf_filter() -> None:
    frame = pd.DataFrame({"VAF": [0.1, 0.2, 0.4], "selected": [True, False, True]})

    result = run_variant_filtering_execution(
        VariantFilteringExecutionInput(
            family="variant_filtering",
            operation="variant_fraction",
            data=frame,
            vaf_column="VAF",
            vaf_max=0.3,
            effect_column="selected",
            return_format="proportion",
            decimal_places=2,
        )
    )

    assert result.answer == "0.50"


def test_gene_with_max_variants() -> None:
    frame = pd.DataFrame({"gene": ["NOTCH1", "NOTCH1", "FLT3"]})

    result = run_variant_filtering_execution(
        VariantFilteringExecutionInput(
            family="variant_filtering",
            operation="gene_with_max_variants",
            data=frame,
            gene_column="gene",
            return_format="label",
        )
    )

    assert result.answer == "NOTCH1"


def test_sample_specific_count() -> None:
    frame = pd.DataFrame({"sample": ["a", "a", "b"]})

    result = run_variant_filtering_execution(
        VariantFilteringExecutionInput(
            family="variant_filtering",
            operation="sample_variant_count",
            data=frame,
            sample_column="sample",
            sample_value="a",
            return_format="number",
        )
    )

    assert result.answer == "2"


def test_percentage_formatting() -> None:
    frame = pd.DataFrame({"selected": [True, False, True, False], "VAF": [0.1] * 4})

    result = run_variant_filtering_execution(
        VariantFilteringExecutionInput(
            family="variant_filtering",
            operation="variant_proportion",
            data=frame,
            effect_column="selected",
            vaf_column="VAF",
            vaf_max=0.3,
            return_format="percentage",
        )
    )

    assert result.answer == "50"


def test_vaf_filtering_coerces_numeric_strings() -> None:
    frame = pd.DataFrame({"VAF": ["0.1", "0.2", "not-a-number"]})

    result = run_variant_filtering_execution(
        VariantFilteringExecutionInput(
            family="variant_filtering",
            operation="filtered_variant_count",
            data=frame,
            vaf_column="VAF",
            vaf_max=0.3,
            return_format="number",
        )
    )

    assert result.answer == "2"
