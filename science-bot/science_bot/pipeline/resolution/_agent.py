"""Iterative controller for the resolution stage."""

from pathlib import Path
from typing import cast

import pandas as pd
from pydantic import TypeAdapter

from science_bot.pipeline.contracts import QuestionFamily
from science_bot.pipeline.execution.schemas import (
    AggregateExecutionInput,
    DifferentialExpressionExecutionInput,
    ExecutionPayload,
    HypothesisTestExecutionInput,
    RegressionExecutionInput,
    ResolvedFilter,
    VariantFilteringExecutionInput,
)
from science_bot.pipeline.resolution._prompts import (
    build_resolution_prompt,
    build_system_prompt,
)
from science_bot.pipeline.resolution._summaries import (
    shortlist_candidate_files,
    summarize_discovery,
    summarize_finalize,
    summarize_tool_result,
    update_scratchpad_from_tool_result,
)
from science_bot.pipeline.resolution.schemas import (
    AggregateResolvedPlan,
    DifferentialExpressionResolvedPlan,
    FamilyResolutionPlan,
    HypothesisTestResolvedPlan,
    RegressionResolvedPlan,
    ResolutionDecision,
    ResolutionScratchpad,
    ResolutionStageInput,
    ResolutionStageOutput,
    ResolvedFilterPlan,
    VariantFilteringResolvedPlan,
)
from science_bot.pipeline.resolution.tools import (
    find_files_with_column,
    get_column_stats,
    get_column_values,
    get_file_schema,
    get_row_sample,
    list_capsule_files,
    list_excel_sheets,
    list_zip_contents,
    load_dataframe,
    search_column_for_value,
    search_columns,
)
from science_bot.providers import parse_structured

MAX_RESOLUTION_ITERATIONS = 8
_PLAN_ADAPTER: TypeAdapter[FamilyResolutionPlan] = TypeAdapter(FamilyResolutionPlan)


class ResolutionError(Exception):
    """Base exception for resolution-stage failures."""


class ResolutionIterationLimitError(ResolutionError):
    """Raised when the resolver fails to converge within the step budget."""


class ResolutionValidationError(ResolutionError):
    """Raised when a plan or tool action is invalid."""


def _initial_scratchpad(stage_input: ResolutionStageInput) -> ResolutionScratchpad:
    manifest = list_capsule_files(stage_input.capsule_path)
    candidates = shortlist_candidate_files(manifest, stage_input.classification.family)
    return ResolutionScratchpad(
        family=stage_input.classification.family,
        question=stage_input.question,
        candidate_files=candidates,
    )


async def run_resolution_agent(
    stage_input: ResolutionStageInput,
) -> ResolutionStageOutput:
    """Resolve a question into a concrete execution payload.

    Args:
        stage_input: Validated resolution stage input.

    Returns:
        ResolutionStageOutput: Final execution payload and compact debug steps.

    Raises:
        ResolutionIterationLimitError: If the controller exceeds its step budget.
        ResolutionValidationError: If the LLM emits an invalid action or plan.
        ResolutionError: If the LLM explicitly fails resolution.
    """
    scratchpad = _initial_scratchpad(stage_input)
    steps = [summarize_discovery(scratchpad, step_index=1)]
    notes: list[str] = []
    previous_signature: tuple[str, tuple[tuple[str, object], ...]] | None = None

    for iteration in range(MAX_RESOLUTION_ITERATIONS):
        scratchpad.iterations_used = iteration
        decision = await parse_structured(
            system_prompt=build_system_prompt(stage_input.classification.family),
            user_prompt=build_resolution_prompt(
                question=stage_input.question,
                scratchpad=scratchpad,
                iterations_remaining=MAX_RESOLUTION_ITERATIONS - iteration,
            ),
            response_model=ResolutionDecision,
        )

        if decision.kind == "fail":
            raise ResolutionError(decision.reason)

        if decision.kind == "use_tool":
            signature = _tool_signature(decision.tool_name or "", decision.arguments)
            if signature == previous_signature:
                raise ResolutionIterationLimitError(
                    "Resolution repeated the same tool call without new progress."
                )
            previous_signature = signature
            result = _execute_tool(
                capsule_path=stage_input.capsule_path,
                tool_name=decision.tool_name or "",
                arguments=decision.arguments,
            )
            _capture_tool_specific_state(
                scratchpad=scratchpad,
                tool_name=decision.tool_name or "",
                arguments=decision.arguments,
                result=result,
            )
            update_scratchpad_from_tool_result(
                scratchpad=scratchpad,
                tool_name=decision.tool_name or "",
                result=result,
            )
            steps.append(
                summarize_tool_result(
                    tool_name=decision.tool_name or "",
                    result=result,
                    scratchpad=scratchpad,
                    step_index=len(steps) + 1,
                )
            )
            continue

        plan = _validate_plan(decision.plan_payload, stage_input.classification.family)
        payload, selected_files, extra_notes = _assemble_payload(
            capsule_path=stage_input.capsule_path,
            plan=plan,
        )
        notes.extend(extra_notes)
        steps.append(
            summarize_finalize(
                family=stage_input.classification.family,
                selected_files=selected_files,
                resolved_field_keys=list(plan.model_dump(exclude_none=True).keys()),
                step_index=len(steps) + 1,
            )
        )
        return ResolutionStageOutput(
            payload=payload,
            iterations_used=iteration + 1,
            selected_files=selected_files,
            notes=notes,
            steps=steps,
        )

    raise ResolutionIterationLimitError(
        f"Resolution exceeded {MAX_RESOLUTION_ITERATIONS} iterations."
    )


def _execute_tool(
    *,
    capsule_path: Path,
    tool_name: str,
    arguments: dict[str, object],
) -> object:
    """Execute one allowed inspection tool.

    Args:
        capsule_path: Capsule directory under analysis.
        tool_name: Allowed tool name chosen by the LLM.
        arguments: Raw structured tool arguments.

    Returns:
        object: Tool result object.

    Raises:
        ResolutionValidationError: If arguments are missing or the tool is unknown.
    """
    try:
        if tool_name == "list_zip_contents":
            return list_zip_contents(
                capsule_path,
                zip_filename=_expect_str(arguments, "zip_filename"),
            )
        if tool_name == "list_excel_sheets":
            return list_excel_sheets(
                capsule_path,
                filename=_expect_str(arguments, "filename"),
            )
        if tool_name == "find_files_with_column":
            return find_files_with_column(
                capsule_path,
                query=_expect_str(arguments, "query"),
            )
        if tool_name == "get_file_schema":
            return get_file_schema(
                capsule_path,
                filename=_expect_str(arguments, "filename"),
            )
        if tool_name == "search_columns":
            return search_columns(
                capsule_path,
                filename=_expect_str(arguments, "filename"),
                query=_expect_str(arguments, "query"),
            )
        if tool_name == "get_column_values":
            return get_column_values(
                capsule_path,
                filename=_expect_str(arguments, "filename"),
                column=_expect_str(arguments, "column"),
                max_values=_expect_int(arguments, "max_values", default=50),
            )
        if tool_name == "get_column_stats":
            return get_column_stats(
                capsule_path,
                filename=_expect_str(arguments, "filename"),
                column=_expect_str(arguments, "column"),
            )
        if tool_name == "search_column_for_value":
            return search_column_for_value(
                capsule_path,
                filename=_expect_str(arguments, "filename"),
                column=_expect_str(arguments, "column"),
                query=_expect_str(arguments, "query"),
                max_matches=_expect_int(arguments, "max_matches", default=50),
            )
        if tool_name == "get_row_sample":
            return get_row_sample(
                capsule_path,
                filename=_expect_str(arguments, "filename"),
                columns=_expect_str_list(arguments, "columns"),
                n=_expect_int(arguments, "n", default=10),
                random_sample=_expect_bool(arguments, "random_sample", default=False),
            )
        raise ResolutionValidationError(f"Unsupported tool requested: {tool_name}")
    except TypeError as exc:
        raise ResolutionValidationError(
            f"Invalid arguments for {tool_name}: {arguments}"
        ) from exc


def _capture_tool_specific_state(
    *,
    scratchpad: ResolutionScratchpad,
    tool_name: str,
    arguments: dict[str, object],
    result: object,
) -> None:
    """Capture state that depends on both tool arguments and results.

    Args:
        scratchpad: Scratchpad to mutate.
        tool_name: Executed tool name.
        arguments: Structured tool arguments.
        result: Tool result object.
    """
    if tool_name == "list_excel_sheets":
        filename = cast(str, arguments.get("filename"))
        if isinstance(result, list) and all(isinstance(item, str) for item in result):
            scratchpad.known_sheets[filename] = cast(list[str], result[:20])
            if filename not in scratchpad.selected_files:
                scratchpad.selected_files.append(filename)
    if tool_name == "find_files_with_column" and isinstance(result, list):
        for item in result[:20]:
            if (
                hasattr(item, "filename")
                and isinstance(item.filename, str)
                and item.filename not in scratchpad.selected_files
            ):
                scratchpad.selected_files.append(item.filename)


def _tool_signature(
    tool_name: str,
    arguments: dict[str, object],
) -> tuple[str, tuple[tuple[str, object], ...]]:
    items = tuple(sorted(arguments.items(), key=lambda item: item[0]))
    return (tool_name, items)


def _expect_str(arguments: dict[str, object], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str):
        raise ResolutionValidationError(f"{key} must be a string.")
    return value


def _expect_int(arguments: dict[str, object], key: str, *, default: int) -> int:
    value = arguments.get(key, default)
    if not isinstance(value, int):
        raise ResolutionValidationError(f"{key} must be an integer.")
    return value


def _expect_bool(arguments: dict[str, object], key: str, *, default: bool) -> bool:
    value = arguments.get(key, default)
    if not isinstance(value, bool):
        raise ResolutionValidationError(f"{key} must be a boolean.")
    return value


def _expect_str_list(arguments: dict[str, object], key: str) -> list[str]:
    value = arguments.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ResolutionValidationError(f"{key} must be a list of strings.")
    return cast(list[str], value)


def _validate_plan(
    payload: dict[str, object] | None,
    family: QuestionFamily,
) -> FamilyResolutionPlan:
    """Validate a final plan payload and enforce the selected family.

    Args:
        payload: Raw plan payload returned by the LLM.
        family: Supported family selected by classification.

    Returns:
        FamilyResolutionPlan: Validated family-specific plan.

    Raises:
        ResolutionValidationError: If the plan is missing or mismatched.
    """
    if payload is None:
        raise ResolutionValidationError("finalize did not include a plan payload.")
    plan = _PLAN_ADAPTER.validate_python(payload)
    if plan.family != family:
        raise ResolutionValidationError(
            f"finalize plan family {plan.family!r} does not match {family!r}."
        )
    return plan


def _assemble_payload(
    *,
    capsule_path: Path,
    plan: FamilyResolutionPlan,
) -> tuple[ExecutionPayload, list[str], list[str]]:
    """Convert a validated resolved plan into an execution payload.

    Args:
        capsule_path: Capsule directory under analysis.
        plan: Validated resolved plan.

    Returns:
        tuple[ExecutionPayload, list[str], list[str]]: Payload, selected files,
        and run-level notes.

    Raises:
        ResolutionValidationError: If plan materialization would be unsafe.
    """
    if isinstance(plan, AggregateResolvedPlan):
        required = _ordered_columns(
            plan.value_column,
            plan.numerator_mask_column,
            plan.denominator_mask_column,
            *_filter_columns(plan.filters),
            *_filter_columns(plan.numerator_filters),
            *_filter_columns(plan.denominator_filters),
        )
        data, notes = _load_required_dataframe(
            capsule_path=capsule_path,
            filename=plan.filename,
            required_columns=required,
        )
        payload = AggregateExecutionInput(
            family=plan.family,
            operation=plan.operation,
            data=data,
            value_column=plan.value_column,
            numerator_mask_column=plan.numerator_mask_column,
            denominator_mask_column=plan.denominator_mask_column,
            numerator_filters=_to_execution_filters(plan.numerator_filters),
            denominator_filters=_to_execution_filters(plan.denominator_filters),
            filters=_to_execution_filters(plan.filters),
            return_format=plan.return_format,
            decimal_places=plan.decimal_places,
            round_to=plan.round_to,
        )
        return payload, [plan.filename], notes

    if isinstance(plan, HypothesisTestResolvedPlan):
        required = _ordered_columns(
            plan.value_column,
            plan.second_value_column,
            plan.group_column,
            *_filter_columns(plan.filters),
        )
        data, notes = _load_required_dataframe(
            capsule_path=capsule_path,
            filename=plan.filename,
            required_columns=required,
        )
        payload = HypothesisTestExecutionInput(
            family=plan.family,
            test=plan.test,
            data=data,
            value_column=plan.value_column,
            second_value_column=plan.second_value_column,
            group_column=plan.group_column,
            group_a_value=plan.group_a_value,
            group_b_value=plan.group_b_value,
            filters=_to_execution_filters(plan.filters),
            return_field=plan.return_field,
            decimal_places=plan.decimal_places,
            round_to=plan.round_to,
        )
        return payload, [plan.filename], notes

    if isinstance(plan, RegressionResolvedPlan):
        required = _ordered_columns(
            plan.outcome_column,
            plan.predictor_column,
            *plan.covariate_columns,
            *_filter_columns(plan.filters),
            *plan.prediction_inputs.keys(),
        )
        data, notes = _load_required_dataframe(
            capsule_path=capsule_path,
            filename=plan.filename,
            required_columns=required,
        )
        payload = RegressionExecutionInput(
            family=plan.family,
            model_type=plan.model_type,
            data=data,
            outcome_column=plan.outcome_column,
            predictor_column=plan.predictor_column,
            covariate_columns=plan.covariate_columns,
            degree=plan.degree,
            prediction_inputs=plan.prediction_inputs,
            filters=_to_execution_filters(plan.filters),
            return_field=plan.return_field,
            decimal_places=plan.decimal_places,
            round_to=plan.round_to,
        )
        return payload, [plan.filename], notes

    if isinstance(plan, DifferentialExpressionResolvedPlan):
        if plan.mode != "precomputed_results":
            raise ResolutionValidationError(
                "Only precomputed_results differential expression mode is supported."
            )
        result_tables = {}
        notes: list[str] = []
        required = _ordered_columns(
            plan.gene_column,
            plan.log_fold_change_column,
            plan.adjusted_p_value_column,
            plan.base_mean_column,
        )
        for label, filename in plan.result_table_files.items():
            data, file_notes = _load_required_dataframe(
                capsule_path=capsule_path,
                filename=filename,
                required_columns=required,
            )
            result_tables[label] = data
            notes.extend(file_notes)
        payload = DifferentialExpressionExecutionInput(
            family=plan.family,
            mode=plan.mode,
            operation=plan.operation,
            result_tables=result_tables,
            comparison_labels=plan.comparison_labels,
            reference_label=plan.reference_label,
            target_gene=plan.target_gene,
            gene_column=plan.gene_column,
            log_fold_change_column=plan.log_fold_change_column,
            adjusted_p_value_column=plan.adjusted_p_value_column,
            base_mean_column=plan.base_mean_column,
            significance_threshold=plan.significance_threshold,
            fold_change_threshold=plan.fold_change_threshold,
            basemean_threshold=plan.basemean_threshold,
            use_lfc_shrinkage=plan.use_lfc_shrinkage,
            correction_methods=plan.correction_methods,
            decimal_places=plan.decimal_places,
            round_to=plan.round_to,
        )
        return payload, list(plan.result_table_files.values()), notes

    if isinstance(plan, VariantFilteringResolvedPlan):
        required = _ordered_columns(
            plan.sample_column,
            plan.gene_column,
            plan.effect_column,
            plan.vaf_column,
            *_filter_columns(plan.filters),
        )
        data, notes = _load_required_dataframe(
            capsule_path=capsule_path,
            filename=plan.filename,
            required_columns=required,
        )
        payload = VariantFilteringExecutionInput(
            family=plan.family,
            operation=plan.operation,
            data=data,
            sample_column=plan.sample_column,
            sample_value=plan.sample_value,
            gene_column=plan.gene_column,
            effect_column=plan.effect_column,
            vaf_column=plan.vaf_column,
            vaf_min=plan.vaf_min,
            vaf_max=plan.vaf_max,
            filters=_to_execution_filters(plan.filters),
            return_format=plan.return_format,
            decimal_places=plan.decimal_places,
            round_to=plan.round_to,
        )
        return payload, [plan.filename], notes

    raise ResolutionValidationError(f"Unsupported resolved plan type: {type(plan)!r}")


def _load_required_dataframe(
    *,
    capsule_path: Path,
    filename: str,
    required_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Load a dataframe using explicit columns only.

    Args:
        capsule_path: Capsule directory under analysis.
        filename: Concrete filename or extended filename syntax.
        required_columns: Explicit required column names.

    Returns:
        tuple[object, list[str]]: DataFrame and human-readable notes.

    Raises:
        ResolutionValidationError: If no required columns are available.
    """
    if not required_columns:
        raise ResolutionValidationError(
            f"Resolution plan for {filename!r} did not identify required columns."
        )
    notes: list[str] = []
    if len(required_columns) < 200:
        notes.append(
            f"Loaded {filename} with explicit column subset of "
            f"{len(required_columns)} columns."
        )
    data = load_dataframe(capsule_path, filename, columns=required_columns)
    return data, notes


def _ordered_columns(*columns: str | None) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for column in columns:
        if column and column not in seen:
            ordered.append(column)
            seen.add(column)
    return ordered


def _filter_columns(filters: list[ResolvedFilterPlan]) -> list[str]:
    columns: list[str] = []
    for filter_item in filters:
        columns.append(filter_item.column)
    return columns


def _to_execution_filters(filters: list[ResolvedFilterPlan]) -> list[ResolvedFilter]:
    return [
        ResolvedFilter(
            column=filter_item.column,
            operator=filter_item.operator,
            value=filter_item.value,
        )
        for filter_item in filters
    ]
