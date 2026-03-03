"""Shared family schemas used across pipeline stages."""

from collections.abc import Mapping
from typing import Annotated, Literal, TypeAlias, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

QuestionFamily: TypeAlias = Literal[
    "aggregate",
    "hypothesis_test",
    "regression",
    "differential_expression",
    "variant_filtering",
]
AggregateOperation: TypeAlias = Literal[
    "count",
    "mean",
    "median",
    "variance",
    "skewness",
    "percentage",
    "proportion",
    "ratio",
]
AggregateReturnFormat: TypeAlias = Literal[
    "number",
    "percentage",
    "ratio",
    "label",
]
HypothesisTestType: TypeAlias = Literal[
    "t_test",
    "mann_whitney_u",
    "chi_square",
    "shapiro_wilk",
    "pearson_correlation",
    "cohens_d",
]
HypothesisTestReturnField: TypeAlias = Literal[
    "statistic",
    "p_value",
    "w_statistic",
    "u_statistic",
    "correlation",
    "effect_size",
]
RegressionModelType: TypeAlias = Literal[
    "logistic",
    "ordinal_logistic",
    "linear",
    "polynomial",
]
RegressionReturnField: TypeAlias = Literal[
    "odds_ratio",
    "percent_increase_in_odds",
    "predicted_probability",
    "coefficient",
    "r_squared",
]
DifferentialExpressionMethod: TypeAlias = Literal[
    "deseq2_like",
    "precomputed_de_table",
]
DifferentialExpressionOperation: TypeAlias = Literal[
    "significant_gene_count",
    "unique_significant_gene_count",
    "shared_overlap_pattern",
    "gene_log2_fold_change",
    "significant_marker_count",
    "correction_ratio",
]
DifferentialExpressionReturnFormat: TypeAlias = Literal[
    "number",
    "ratio",
    "label",
    "signed_number",
]
VariantFilteringOperation: TypeAlias = Literal[
    "filtered_variant_count",
    "variant_fraction",
    "variant_proportion",
    "gene_with_max_variants",
    "sample_variant_count",
]
VariantFilteringReturnFormat: TypeAlias = Literal[
    "number",
    "proportion",
    "percentage",
    "label",
]
ScalarValue: TypeAlias = str | int | float | bool


class SupportedQuestionClassification(BaseModel):
    """Structured classification for a supported question family."""

    model_config = ConfigDict(extra="forbid")

    family: QuestionFamily


class UnsupportedQuestionClassification(BaseModel):
    """Structured classification for a question outside current support."""

    model_config = ConfigDict(extra="forbid")

    family: Literal["unsupported"]
    reason: str

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        """Validate the unsupported reason field.

        Args:
            value: Unsupported reason text.

        Returns:
            str: Stripped unsupported reason text.

        Raises:
            ValueError: If the reason is empty.
        """
        stripped = value.strip()
        if not stripped:
            raise ValueError(
                "Unsupported question classification requires a non-empty reason."
            )
        return stripped


QuestionClassification: TypeAlias = Annotated[
    SupportedQuestionClassification | UnsupportedQuestionClassification,
    Field(discriminator="family"),
]

QuestionClassificationAdapter: TypeAdapter[QuestionClassification] = TypeAdapter(
    QuestionClassification
)


def parse_question_classification(payload: object) -> QuestionClassification:
    """Parse a raw payload into a question classification.

    Args:
        payload: Raw payload to validate.

    Returns:
        QuestionClassification: Parsed supported or unsupported classification.
    """
    if isinstance(payload, Mapping):
        mapping_payload = cast(Mapping[str, object], payload)
        family = mapping_payload.get("family")
        if family != "unsupported" and "reason" in mapping_payload:
            payload = {
                key: value for key, value in mapping_payload.items() if key != "reason"
            }
    return QuestionClassificationAdapter.validate_python(payload)


class FilterCondition(BaseModel):
    """Hint-based filter condition extracted from a question."""

    model_config = ConfigDict(extra="forbid")

    field_hint: str
    operator: Literal["==", "!=", ">", ">=", "<", "<=", "in", "contains"]
    value: ScalarValue


class PredictionInputItem(BaseModel):
    """Structured prediction input hint for regression questions."""

    model_config = ConfigDict(extra="forbid")

    field_hint: str
    value: ScalarValue


class AggregateQuestionSpec(BaseModel):
    """Structured aggregate question specification."""

    model_config = ConfigDict(extra="forbid")

    family: Literal["aggregate"]
    operation: AggregateOperation
    value_field_hint: str | None = None
    group_field_hint: str | None = None
    entity_hint: str | None = None
    filters: list[FilterCondition] = Field(default_factory=list)
    return_format: AggregateReturnFormat
    round_to: int | None = None

    @model_validator(mode="after")
    def validate_aggregate_fields(self) -> "AggregateQuestionSpec":
        """Validate aggregate-specific field requirements.

        Returns:
            AggregateQuestionSpec: Validated aggregate question spec.

        Raises:
            ValueError: If required hints are missing for the selected operation.
        """
        if (
            self.operation in {"mean", "median", "variance", "skewness"}
            and not self.value_field_hint
        ):
            raise ValueError(
                "Aggregate operations mean, median, variance, and skewness require "
                "value_field_hint."
            )
        return self


class HypothesisTestQuestionSpec(BaseModel):
    """Structured hypothesis test question specification."""

    model_config = ConfigDict(extra="forbid")

    family: Literal["hypothesis_test"]
    test: HypothesisTestType
    value_field_hint: str | None = None
    second_value_field_hint: str | None = None
    group_field_hint: str | None = None
    group_a_hint: str | None = None
    group_b_hint: str | None = None
    filters: list[FilterCondition] = Field(default_factory=list)
    return_field: HypothesisTestReturnField
    round_to: int | None = None

    @model_validator(mode="after")
    def validate_hypothesis_test_fields(self) -> "HypothesisTestQuestionSpec":
        """Validate hypothesis-test-specific field requirements.

        Returns:
            HypothesisTestQuestionSpec: Validated hypothesis test spec.

        Raises:
            ValueError: If required hints are missing for the selected test.
        """
        if self.test == "pearson_correlation" and not self.second_value_field_hint:
            raise ValueError("pearson_correlation requires second_value_field_hint.")
        if self.test in {"t_test", "mann_whitney_u", "chi_square", "cohens_d"}:
            if not self.group_field_hint:
                raise ValueError(
                    f"{self.test} requires group_field_hint for group comparison."
                )
        if self.test == "shapiro_wilk" and self.group_field_hint is not None:
            raise ValueError("shapiro_wilk should not include group_field_hint.")
        return self


class RegressionQuestionSpec(BaseModel):
    """Structured regression question specification."""

    model_config = ConfigDict(extra="forbid")

    family: Literal["regression"]
    model_type: RegressionModelType
    outcome_field_hint: str
    predictor_field_hint: str
    covariate_field_hints: list[str] = Field(default_factory=list)
    degree: int | None = None
    prediction_inputs: list[PredictionInputItem] = Field(default_factory=list)
    filters: list[FilterCondition] = Field(default_factory=list)
    return_field: RegressionReturnField
    round_to: int | None = None

    @model_validator(mode="after")
    def validate_regression_fields(self) -> "RegressionQuestionSpec":
        """Validate regression-specific field requirements.

        Returns:
            RegressionQuestionSpec: Validated regression question spec.

        Raises:
            ValueError: If selected options are inconsistent.
        """
        if self.model_type == "polynomial":
            if self.degree is None:
                raise ValueError("polynomial regression requires degree.")
        elif self.degree is not None:
            raise ValueError("Only polynomial regression may set degree.")
        if self.return_field == "predicted_probability" and not self.prediction_inputs:
            raise ValueError("predicted_probability requires prediction_inputs.")
        return self


class DifferentialExpressionQuestionSpec(BaseModel):
    """Structured differential expression question specification."""

    model_config = ConfigDict(extra="forbid")

    family: Literal["differential_expression"]
    method: DifferentialExpressionMethod
    operation: DifferentialExpressionOperation
    comparison_label_hints: list[str] = Field(default_factory=list)
    reference_label_hint: str | None = None
    target_gene_hint: str | None = None
    significance_threshold: float | None = None
    fold_change_threshold: float | None = None
    basemean_threshold: float | None = None
    use_lfc_shrinkage: bool = False
    correction_method_hints: list[str] = Field(default_factory=list)
    filters: list[FilterCondition] = Field(default_factory=list)
    return_format: DifferentialExpressionReturnFormat
    round_to: int | None = None

    @model_validator(mode="after")
    def validate_differential_expression_fields(
        self,
    ) -> "DifferentialExpressionQuestionSpec":
        """Validate differential-expression-specific field requirements.

        Returns:
            DifferentialExpressionQuestionSpec: Validated differential expression spec.

        Raises:
            ValueError: If required hints are missing.
        """
        if self.operation == "gene_log2_fold_change" and not self.target_gene_hint:
            raise ValueError("gene_log2_fold_change requires target_gene_hint.")
        if (
            self.operation == "shared_overlap_pattern"
            and len(self.comparison_label_hints) < 2
        ):
            raise ValueError(
                "shared_overlap_pattern requires at least two comparison labels."
            )
        return self


class VariantFilteringQuestionSpec(BaseModel):
    """Structured variant filtering question specification."""

    model_config = ConfigDict(extra="forbid")

    family: Literal["variant_filtering"]
    operation: VariantFilteringOperation
    cohort_hint: str | None = None
    sample_hint: str | None = None
    gene_hint: str | None = None
    effect_hint: str | None = None
    pathogenicity_hint: str | None = None
    vaf_min: float | None = None
    vaf_max: float | None = None
    filters: list[FilterCondition] = Field(default_factory=list)
    return_format: VariantFilteringReturnFormat
    round_to: int | None = None

    @model_validator(mode="after")
    def validate_variant_filtering_fields(self) -> "VariantFilteringQuestionSpec":
        """Validate variant-filtering-specific field requirements.

        Returns:
            VariantFilteringQuestionSpec: Validated variant filtering spec.

        Raises:
            ValueError: If VAF bounds are inconsistent.
        """
        if (
            self.vaf_min is not None
            and self.vaf_max is not None
            and self.vaf_min > self.vaf_max
        ):
            raise ValueError("vaf_min must be less than or equal to vaf_max.")
        return self


QuestionExecutionSpec: TypeAlias = Annotated[
    AggregateQuestionSpec
    | HypothesisTestQuestionSpec
    | RegressionQuestionSpec
    | DifferentialExpressionQuestionSpec
    | VariantFilteringQuestionSpec,
    Field(discriminator="family"),
]

QuestionExecutionSpecAdapter: TypeAdapter[QuestionExecutionSpec] = TypeAdapter(
    QuestionExecutionSpec
)


def parse_question_execution_spec(payload: object) -> QuestionExecutionSpec:
    """Parse a raw payload into a supported family question spec.

    Args:
        payload: Raw payload to validate.

    Returns:
        QuestionExecutionSpec: Parsed supported family question spec.
    """
    return QuestionExecutionSpecAdapter.validate_python(payload)
