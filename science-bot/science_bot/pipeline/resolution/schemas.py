"""Schemas local to the resolution stage."""

from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from science_bot.pipeline.contracts import (
    AggregateOperation,
    AggregateReturnFormat,
    DifferentialExpressionOperation,
    HypothesisTestReturnField,
    HypothesisTestType,
    QuestionFamily,
    RegressionModelType,
    RegressionReturnField,
    ScalarValue,
    SupportedQuestionClassification,
    VariantFilteringOperation,
    VariantFilteringReturnFormat,
)
from science_bot.pipeline.execution.schemas import ExecutionPayload

ResolvedFilterValue: TypeAlias = (
    str | int | float | bool | list[str] | list[int] | list[float]
)


class ResolutionStepSummary(BaseModel):
    """Compact summary of one resolver step."""

    model_config = ConfigDict(extra="forbid")

    step_index: int
    kind: Literal["discover", "tool", "finalize"]
    tool_name: str | None = None
    message: str
    selected_files: list[str] = Field(default_factory=list)
    resolved_field_keys: list[str] = Field(default_factory=list)
    truncated: bool = False

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        """Validate the human-readable step summary message.

        Args:
            value: Candidate message text.

        Returns:
            str: Stripped message text.

        Raises:
            ValueError: If the message is empty.
        """
        stripped = value.strip()
        if not stripped:
            raise ValueError("message must be non-empty.")
        return stripped


class ResolutionStageInput(BaseModel):
    """Input contract for the resolution stage."""

    model_config = ConfigDict(extra="forbid")

    question: str
    classification: SupportedQuestionClassification
    capsule_path: Path

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        """Validate the question text.

        Args:
            value: Candidate question text.

        Returns:
            str: Stripped question text.

        Raises:
            ValueError: If the question is empty.
        """
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must be non-empty.")
        return stripped

    @field_validator("capsule_path")
    @classmethod
    def validate_capsule_path(cls, value: Path) -> Path:
        """Validate the capsule path.

        Args:
            value: Candidate capsule directory.

        Returns:
            Path: Resolved capsule path.

        Raises:
            ValueError: If the path does not exist.
        """
        resolved = value.expanduser().resolve()
        if not resolved.exists():
            raise ValueError(f"capsule_path does not exist: {resolved}")
        return resolved


class ResolutionStageOutput(BaseModel):
    """Output contract for the resolution stage."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    payload: ExecutionPayload
    iterations_used: int
    selected_files: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    steps: list[ResolutionStepSummary] = Field(default_factory=list)


class ResolvedFilterPlan(BaseModel):
    """Resolved filter ready to convert into execution-stage filters."""

    model_config = ConfigDict(extra="forbid")

    column: str
    operator: Literal["==", "!=", ">", ">=", "<", "<=", "in", "contains"]
    value: ResolvedFilterValue


class AggregateResolvedPlan(BaseModel):
    """Resolved aggregate plan referencing concrete file and column names."""

    model_config = ConfigDict(extra="forbid")

    family: Literal["aggregate"] = "aggregate"
    filename: str
    operation: AggregateOperation
    value_column: str | None = None
    numerator_mask_column: str | None = None
    denominator_mask_column: str | None = None
    numerator_filters: list[ResolvedFilterPlan] = Field(default_factory=list)
    denominator_filters: list[ResolvedFilterPlan] = Field(default_factory=list)
    filters: list[ResolvedFilterPlan] = Field(default_factory=list)
    return_format: AggregateReturnFormat
    decimal_places: int | None = None
    round_to: int | None = None

    @model_validator(mode="after")
    def validate_plan(self) -> "AggregateResolvedPlan":
        """Validate aggregate-specific resolved fields.

        Returns:
            AggregateResolvedPlan: Validated plan.

        Raises:
            ValueError: If required columns or filters are missing.
        """
        if (
            self.operation in {"mean", "median", "variance", "skewness"}
            and not self.value_column
        ):
            raise ValueError(
                "mean, median, variance, and skewness require value_column."
            )
        if self.operation in {"percentage", "proportion", "ratio"}:
            if not self.numerator_mask_column and not self.numerator_filters:
                raise ValueError(
                    "percentage, proportion, and ratio require numerator mask "
                    "or numerator filters."
                )
        if self.operation == "ratio":
            if not self.denominator_mask_column and not self.denominator_filters:
                raise ValueError(
                    "ratio requires denominator mask or denominator filters."
                )
        return self


class HypothesisTestResolvedPlan(BaseModel):
    """Resolved hypothesis-test plan referencing concrete file and columns."""

    model_config = ConfigDict(extra="forbid")

    family: Literal["hypothesis_test"] = "hypothesis_test"
    filename: str
    test: HypothesisTestType
    value_column: str | None = None
    second_value_column: str | None = None
    group_column: str | None = None
    group_a_value: ScalarValue | None = None
    group_b_value: ScalarValue | None = None
    filters: list[ResolvedFilterPlan] = Field(default_factory=list)
    return_field: HypothesisTestReturnField
    decimal_places: int | None = None
    round_to: int | None = None

    @model_validator(mode="after")
    def validate_plan(self) -> "HypothesisTestResolvedPlan":
        """Validate hypothesis-test-specific resolved fields.

        Returns:
            HypothesisTestResolvedPlan: Validated plan.

        Raises:
            ValueError: If required columns or labels are missing.
        """
        if self.test == "pearson_correlation":
            if not self.value_column or not self.second_value_column:
                raise ValueError(
                    "pearson_correlation requires value_column and second_value_column."
                )
        elif self.test == "shapiro_wilk":
            if not self.value_column:
                raise ValueError("shapiro_wilk requires value_column.")
        elif self.test in {"t_test", "mann_whitney_u", "cohens_d"}:
            if not self.value_column or not self.group_column:
                raise ValueError(f"{self.test} requires value_column and group_column.")
            if self.group_a_value is None or self.group_b_value is None:
                raise ValueError(
                    f"{self.test} requires group_a_value and group_b_value."
                )
        elif self.test == "chi_square":
            if not self.value_column or not self.group_column:
                raise ValueError("chi_square requires value_column and group_column.")
        return self


class RegressionResolvedPlan(BaseModel):
    """Resolved regression plan referencing concrete file and columns."""

    model_config = ConfigDict(extra="forbid")

    family: Literal["regression"] = "regression"
    filename: str
    model_type: RegressionModelType
    outcome_column: str
    predictor_column: str
    covariate_columns: list[str] = Field(default_factory=list)
    degree: int | None = None
    prediction_inputs: dict[str, ScalarValue] = Field(default_factory=dict)
    filters: list[ResolvedFilterPlan] = Field(default_factory=list)
    return_field: RegressionReturnField
    decimal_places: int | None = None
    round_to: int | None = None

    @model_validator(mode="after")
    def validate_plan(self) -> "RegressionResolvedPlan":
        """Validate regression-specific resolved fields.

        Returns:
            RegressionResolvedPlan: Validated plan.

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


class DifferentialExpressionResolvedPlan(BaseModel):
    """Resolved differential-expression plan for precomputed results."""

    model_config = ConfigDict(extra="forbid")

    family: Literal["differential_expression"] = "differential_expression"
    mode: Literal["precomputed_results"]
    result_table_files: dict[str, str] = Field(default_factory=dict)
    operation: DifferentialExpressionOperation
    comparison_labels: list[str] = Field(default_factory=list)
    reference_label: str | None = None
    target_gene: str | None = None
    gene_column: str | None = None
    log_fold_change_column: str | None = None
    adjusted_p_value_column: str | None = None
    base_mean_column: str | None = None
    significance_threshold: float | None = None
    fold_change_threshold: float | None = None
    basemean_threshold: float | None = None
    use_lfc_shrinkage: bool = False
    correction_methods: list[str] = Field(default_factory=list)
    decimal_places: int | None = None
    round_to: int | None = None

    @model_validator(mode="after")
    def validate_plan(self) -> "DifferentialExpressionResolvedPlan":
        """Validate differential-expression-specific resolved fields.

        Returns:
            DifferentialExpressionResolvedPlan: Validated plan.

        Raises:
            ValueError: If required files or labels are missing.
        """
        if not self.result_table_files:
            raise ValueError("precomputed_results requires result_table_files.")
        if self.operation == "gene_log2_fold_change" and not self.target_gene:
            raise ValueError("gene_log2_fold_change requires target_gene.")
        if (
            self.operation
            in {
                "shared_overlap_pattern",
                "unique_significant_gene_count",
            }
            and len(self.comparison_labels) < 2
        ):
            raise ValueError(
                f"{self.operation} requires at least two comparison_labels."
            )
        if self.operation == "correction_ratio" and len(self.correction_methods) < 2:
            raise ValueError(
                "correction_ratio requires at least two correction_methods."
            )
        return self


class VariantFilteringResolvedPlan(BaseModel):
    """Resolved variant-filtering plan referencing concrete file and columns."""

    model_config = ConfigDict(extra="forbid")

    family: Literal["variant_filtering"] = "variant_filtering"
    filename: str
    operation: VariantFilteringOperation
    sample_column: str | None = None
    sample_value: ScalarValue | None = None
    gene_column: str | None = None
    effect_column: str | None = None
    vaf_column: str | None = None
    vaf_min: float | None = None
    vaf_max: float | None = None
    filters: list[ResolvedFilterPlan] = Field(default_factory=list)
    return_format: VariantFilteringReturnFormat
    decimal_places: int | None = None
    round_to: int | None = None

    @model_validator(mode="after")
    def validate_plan(self) -> "VariantFilteringResolvedPlan":
        """Validate variant-filtering-specific resolved fields.

        Returns:
            VariantFilteringResolvedPlan: Validated plan.

        Raises:
            ValueError: If required columns or VAF bounds are inconsistent.
        """
        if (
            self.vaf_min is not None
            and self.vaf_max is not None
            and self.vaf_min > self.vaf_max
        ):
            raise ValueError("vaf_min must be less than or equal to vaf_max.")
        if (
            self.vaf_min is not None or self.vaf_max is not None
        ) and not self.vaf_column:
            raise ValueError("VAF filtering requires vaf_column.")
        if self.operation == "gene_with_max_variants" and not self.gene_column:
            raise ValueError("gene_with_max_variants requires gene_column.")
        if self.operation == "sample_variant_count":
            if not self.sample_column or self.sample_value is None:
                raise ValueError(
                    "sample_variant_count requires sample_column and sample_value."
                )
        return self


FamilyResolutionPlan: TypeAlias = Annotated[
    AggregateResolvedPlan
    | HypothesisTestResolvedPlan
    | RegressionResolvedPlan
    | DifferentialExpressionResolvedPlan
    | VariantFilteringResolvedPlan,
    Field(discriminator="family"),
]


class CandidateFileSummary(BaseModel):
    """Compact file candidate kept in scratchpad state."""

    model_config = ConfigDict(extra="forbid")

    filename: str
    file_type: str
    size_human: str
    row_count: int | None = None
    column_count: int | None = None
    is_wide: bool | None = None
    relevance_score: int


class ColumnEvidence(BaseModel):
    """Evidence linking a file and columns to a semantic role."""

    model_config = ConfigDict(extra="forbid")

    filename: str
    columns: list[str]
    reason: str


class ValueEvidence(BaseModel):
    """Evidence linking a file column to representative values."""

    model_config = ConfigDict(extra="forbid")

    filename: str
    column: str
    values: list[str]
    reason: str


class ResolutionScratchpad(BaseModel):
    """Compact structured memory carried across resolution iterations."""

    model_config = ConfigDict(extra="forbid")

    family: QuestionFamily
    question: str
    candidate_files: list[CandidateFileSummary] = Field(default_factory=list)
    selected_files: list[str] = Field(default_factory=list)
    known_sheets: dict[str, list[str]] = Field(default_factory=dict)
    known_columns: dict[str, list[str]] = Field(default_factory=dict)
    column_evidence: list[ColumnEvidence] = Field(default_factory=list)
    value_evidence: list[ValueEvidence] = Field(default_factory=list)
    resolved_fields: dict[str, object] = Field(default_factory=dict)
    open_questions: list[str] = Field(default_factory=list)
    last_tool_name: str | None = None
    last_tool_summary: str | None = None
    iterations_used: int = 0


ResolutionAction: TypeAlias = Literal[
    "use_list_zip_contents",
    "use_list_excel_sheets",
    "use_find_files_with_column",
    "use_get_file_schema",
    "use_search_columns",
    "use_get_column_values",
    "use_get_column_stats",
    "use_search_column_for_value",
    "use_get_row_sample",
    "finalize",
    "fail",
]


class PredictionInputEntry(BaseModel):
    """One regression prediction input value."""

    model_config = ConfigDict(extra="forbid")

    column: str
    value: ScalarValue


class ResultTableFileEntry(BaseModel):
    """One differential-expression comparison label to file mapping."""

    model_config = ConfigDict(extra="forbid")

    label: str
    filename: str


class BaseResolutionDecision(BaseModel):
    """Flat structured decision returned by the resolver LLM."""

    model_config = ConfigDict(extra="forbid")

    action: ResolutionAction
    reason: str
    zip_filename: str | None = None
    filename: str | None = None
    query: str | None = None
    column: str | None = None
    columns: list[str] = Field(default_factory=list)
    n: int = 10
    random_sample: bool = False
    max_values: int = 50
    max_matches: int = 50

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        """Validate the decision rationale.

        Args:
            value: Candidate rationale text.

        Returns:
            str: Stripped rationale.

        Raises:
            ValueError: If the rationale is empty.
        """
        stripped = value.strip()
        if not stripped:
            raise ValueError("reason must be non-empty.")
        return stripped

    @field_validator("zip_filename", "filename", "query", "column")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        """Normalize optional text fields.

        Args:
            value: Candidate optional text.

        Returns:
            str | None: Stripped text or null.
        """
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class AggregateResolutionDecision(BaseResolutionDecision):
    """Flat resolver response for aggregate questions."""

    operation: AggregateOperation | None = None
    value_column: str | None = None
    numerator_mask_column: str | None = None
    denominator_mask_column: str | None = None
    numerator_filters: list[ResolvedFilterPlan] = Field(default_factory=list)
    denominator_filters: list[ResolvedFilterPlan] = Field(default_factory=list)
    filters: list[ResolvedFilterPlan] = Field(default_factory=list)
    return_format: AggregateReturnFormat | None = None
    decimal_places: int | None = None
    round_to: int | None = None


class HypothesisTestResolutionDecision(BaseResolutionDecision):
    """Flat resolver response for hypothesis-test questions."""

    test: HypothesisTestType | None = None
    value_column: str | None = None
    second_value_column: str | None = None
    group_column: str | None = None
    group_a_value: ScalarValue | None = None
    group_b_value: ScalarValue | None = None
    filters: list[ResolvedFilterPlan] = Field(default_factory=list)
    return_field: HypothesisTestReturnField | None = None
    decimal_places: int | None = None
    round_to: int | None = None


class RegressionResolutionDecision(BaseResolutionDecision):
    """Flat resolver response for regression questions."""

    model_type: RegressionModelType | None = None
    outcome_column: str | None = None
    predictor_column: str | None = None
    covariate_columns: list[str] = Field(default_factory=list)
    degree: int | None = None
    prediction_inputs: list[PredictionInputEntry] = Field(default_factory=list)
    filters: list[ResolvedFilterPlan] = Field(default_factory=list)
    return_field: RegressionReturnField | None = None
    decimal_places: int | None = None
    round_to: int | None = None


class DifferentialExpressionResolutionDecision(BaseResolutionDecision):
    """Flat resolver response for differential-expression questions."""

    mode: Literal["precomputed_results"] | None = None
    result_table_files: list[ResultTableFileEntry] = Field(default_factory=list)
    operation: DifferentialExpressionOperation | None = None
    comparison_labels: list[str] = Field(default_factory=list)
    reference_label: str | None = None
    target_gene: str | None = None
    gene_column: str | None = None
    log_fold_change_column: str | None = None
    adjusted_p_value_column: str | None = None
    base_mean_column: str | None = None
    significance_threshold: float | None = None
    fold_change_threshold: float | None = None
    basemean_threshold: float | None = None
    use_lfc_shrinkage: bool = False
    correction_methods: list[str] = Field(default_factory=list)
    decimal_places: int | None = None
    round_to: int | None = None


class VariantFilteringResolutionDecision(BaseResolutionDecision):
    """Flat resolver response for variant-filtering questions."""

    operation: VariantFilteringOperation | None = None
    sample_column: str | None = None
    sample_value: ScalarValue | None = None
    gene_column: str | None = None
    effect_column: str | None = None
    vaf_column: str | None = None
    vaf_min: float | None = None
    vaf_max: float | None = None
    filters: list[ResolvedFilterPlan] = Field(default_factory=list)
    return_format: VariantFilteringReturnFormat | None = None
    decimal_places: int | None = None
    round_to: int | None = None


FamilyResolutionDecisionResponse: TypeAlias = (
    AggregateResolutionDecision
    | HypothesisTestResolutionDecision
    | RegressionResolutionDecision
    | DifferentialExpressionResolutionDecision
    | VariantFilteringResolutionDecision
)
