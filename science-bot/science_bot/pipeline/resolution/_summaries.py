"""Summary helpers for compact resolver state updates."""

from typing import cast

from science_bot.pipeline.contracts import QuestionFamily
from science_bot.pipeline.resolution.schemas import (
    CandidateFileSummary,
    ColumnEvidence,
    ResolutionScratchpad,
    ResolutionStepSummary,
    ValueEvidence,
)
from science_bot.pipeline.resolution.tools.schemas import (
    CapsuleManifest,
    ColumnSearchResult,
    ColumnStats,
    ColumnValues,
    ColumnValueSearchResult,
    FileSchema,
    RowSample,
    ZipManifest,
)

MAX_CANDIDATE_FILES = 15
MAX_KNOWN_COLUMNS = 40
MAX_TOOL_SUMMARY_LENGTH = 800
MAX_STEP_MESSAGE_LENGTH = 200
MAX_ZIP_ENTRIES = 20
MAX_VALUES = 20
MAX_ROWS = 5

_FAMILY_KEYWORDS: dict[QuestionFamily, tuple[str, ...]] = {
    "aggregate": (
        "metadata",
        "phenotype",
        "clinical",
        "expression",
        "covariate",
        "table",
    ),
    "hypothesis_test": (
        "metadata",
        "phenotype",
        "clinical",
        "expression",
        "group",
        "table",
    ),
    "regression": (
        "metadata",
        "phenotype",
        "clinical",
        "covariate",
        "expression",
        "table",
    ),
    "differential_expression": (
        "result",
        "differential",
        "de",
        "marker",
        "gene",
        "xlsx",
        "tsv",
        "zip",
    ),
    "variant_filtering": ("variant", "mutation", "maf", "vaf", "gene", "chip"),
}


def shortlist_candidate_files(
    manifest: CapsuleManifest,
    family: QuestionFamily,
) -> list[CandidateFileSummary]:
    """Reduce a manifest to a family-aware shortlist.

    Args:
        manifest: Full capsule manifest from the tools package.
        family: Supported question family.

    Returns:
        list[CandidateFileSummary]: Ranked candidate files.
    """
    keywords = _FAMILY_KEYWORDS[family]
    candidates: list[CandidateFileSummary] = []
    for info in manifest.files:
        score = 0
        filename_lower = info.filename.lower()
        for keyword in keywords:
            if keyword in filename_lower:
                score += 5
        if info.file_type == "zip":
            score += 2 if family == "differential_expression" else 0
        if info.file_type == "excel":
            score += 3 if family == "differential_expression" else 1
        if info.is_wide:
            score -= 1
        candidates.append(
            CandidateFileSummary(
                filename=info.filename,
                file_type=info.file_type,
                size_human=info.size_human,
                row_count=info.row_count,
                column_count=info.column_count,
                is_wide=info.is_wide,
                relevance_score=score,
            )
        )

    candidates.sort(
        key=lambda candidate: (
            candidate.relevance_score,
            candidate.column_count or -1,
            candidate.row_count or -1,
        ),
        reverse=True,
    )
    return candidates[:MAX_CANDIDATE_FILES]


def summarize_discovery(
    scratchpad: ResolutionScratchpad,
    *,
    step_index: int,
) -> ResolutionStepSummary:
    """Build the initial discovery step summary.

    Args:
        scratchpad: Updated scratchpad after discovery.
        step_index: Step index for output ordering.

    Returns:
        ResolutionStepSummary: Compact discovery summary.
    """
    message = (
        f"Shortlisted {len(scratchpad.candidate_files)} candidate files for "
        f"{scratchpad.family} resolution."
    )
    return ResolutionStepSummary(
        step_index=step_index,
        kind="discover",
        message=_truncate(message, MAX_STEP_MESSAGE_LENGTH),
        selected_files=scratchpad.selected_files,
        resolved_field_keys=sorted(scratchpad.resolved_fields.keys()),
    )


def summarize_tool_result(
    *,
    tool_name: str,
    result: object,
    scratchpad: ResolutionScratchpad,
    step_index: int,
) -> ResolutionStepSummary:
    """Build a compact step summary from one tool result.

    Args:
        tool_name: Tool invoked in the iteration.
        result: Tool result object.
        scratchpad: Scratchpad after updates.
        step_index: Step index for output ordering.

    Returns:
        ResolutionStepSummary: Compact tool summary.
    """
    message, truncated = tool_result_message(tool_name, result)
    return ResolutionStepSummary(
        step_index=step_index,
        kind="tool",
        tool_name=tool_name,
        message=_truncate(message, MAX_STEP_MESSAGE_LENGTH),
        selected_files=scratchpad.selected_files,
        resolved_field_keys=sorted(scratchpad.resolved_fields.keys()),
        truncated=truncated,
    )


def summarize_finalize(
    *,
    family: QuestionFamily,
    selected_files: list[str],
    resolved_field_keys: list[str],
    step_index: int,
) -> ResolutionStepSummary:
    """Build the final step summary.

    Args:
        family: Resolved question family.
        selected_files: Files used in the final payload.
        resolved_field_keys: Resolved field names.
        step_index: Step index for output ordering.

    Returns:
        ResolutionStepSummary: Compact finalize summary.
    """
    message = (
        f"Finalized {family} payload from {len(selected_files)} selected "
        f"file{'s' if len(selected_files) != 1 else ''}."
    )
    return ResolutionStepSummary(
        step_index=step_index,
        kind="finalize",
        message=_truncate(message, MAX_STEP_MESSAGE_LENGTH),
        selected_files=selected_files,
        resolved_field_keys=sorted(resolved_field_keys),
    )


def tool_result_message(tool_name: str, result: object) -> tuple[str, bool]:
    """Create a compact human-readable summary for a tool result.

    Args:
        tool_name: Tool invoked in the iteration.
        result: Tool result object.

    Returns:
        tuple[str, bool]: Message text and whether truncation was required.
    """
    if tool_name == "list_zip_contents" and isinstance(result, ZipManifest):
        entries = [entry.inner_path for entry in result.entries if entry.is_readable]
        truncated = len(entries) > MAX_ZIP_ENTRIES
        shown = entries[:MAX_ZIP_ENTRIES]
        return (
            f"Inspected zip {result.zip_filename}; readable entries: {shown}",
            truncated,
        )

    if tool_name == "list_excel_sheets" and isinstance(result, list):
        truncated = len(result) > MAX_VALUES
        return (f"Excel sheets: {result[:MAX_VALUES]}", truncated)

    if isinstance(result, FileSchema):
        columns = [column.name for column in result.columns[:MAX_KNOWN_COLUMNS]]
        truncated = result.columns_truncated or result.column_count > MAX_KNOWN_COLUMNS
        return (
            f"Schema for {result.filename}: {result.row_count} rows, "
            f"{result.column_count} columns, sample columns={columns}",
            truncated,
        )

    if isinstance(result, ColumnSearchResult):
        truncated = result.total_matches > MAX_VALUES
        return (
            f"Column matches in {result.filename} for {result.query!r}: "
            f"{result.matches[:MAX_VALUES]}",
            truncated,
        )

    if (
        isinstance(result, list)
        and result
        and isinstance(result[0], ColumnSearchResult)
    ):
        search_results = cast(list[ColumnSearchResult], result)
        filenames = [entry.filename for entry in search_results[:MAX_VALUES]]
        truncated = len(result) > MAX_VALUES
        return (
            f"Files with matching columns: {filenames}",
            truncated,
        )

    if isinstance(result, ColumnValues):
        values = result.values[:MAX_VALUES]
        return (
            f"Observed values for {result.filename}:{result.column} -> {values}",
            result.truncated or result.unique_count > MAX_VALUES,
        )

    if isinstance(result, ColumnStats):
        if result.most_common is not None:
            return (
                f"Stats for {result.filename}:{result.column} -> "
                f"most_common={result.most_common[:MAX_VALUES]}",
                len(result.most_common) > MAX_VALUES,
            )
        return (
            f"Stats for {result.filename}:{result.column} -> "
            f"min={result.min}, max={result.max}, mean={result.mean}, std={result.std}",
            False,
        )

    if isinstance(result, ColumnValueSearchResult):
        return (
            f"Matching values for {result.filename}:{result.column} -> "
            f"{result.matches[:MAX_VALUES]}",
            result.truncated or result.total_matches > MAX_VALUES,
        )

    if isinstance(result, RowSample):
        truncated = len(result.rows) > MAX_ROWS
        return (
            f"Row sample for {result.filename} using columns {result.columns}: "
            f"{result.rows[:MAX_ROWS]}",
            truncated,
        )

    message = f"Completed {tool_name}."
    return (message, False)


def update_scratchpad_from_tool_result(
    *,
    scratchpad: ResolutionScratchpad,
    tool_name: str,
    result: object,
) -> None:
    """Update scratchpad state from one tool result.

    Args:
        scratchpad: Scratchpad to mutate in place.
        tool_name: Tool name that produced the result.
        result: Tool result object.
    """
    scratchpad.last_tool_name = tool_name
    tool_summary, truncated = tool_result_message(tool_name, result)
    scratchpad.last_tool_summary = _truncate(tool_summary, MAX_TOOL_SUMMARY_LENGTH)

    if isinstance(result, FileSchema):
        scratchpad.known_columns[result.filename] = [
            column.name for column in result.columns[:MAX_KNOWN_COLUMNS]
        ]
        if result.filename not in scratchpad.selected_files:
            scratchpad.selected_files.append(result.filename)
    elif isinstance(result, ColumnSearchResult):
        if result.filename not in scratchpad.selected_files:
            scratchpad.selected_files.append(result.filename)
        scratchpad.column_evidence.append(
            ColumnEvidence(
                filename=result.filename,
                columns=result.matches[:MAX_VALUES],
                reason=f"Matches query {result.query!r}",
            )
        )
        scratchpad.column_evidence = scratchpad.column_evidence[-20:]
    elif isinstance(result, ColumnValues):
        if result.filename not in scratchpad.selected_files:
            scratchpad.selected_files.append(result.filename)
        scratchpad.value_evidence.append(
            ValueEvidence(
                filename=result.filename,
                column=result.column,
                values=[str(value) for value in result.values[:MAX_VALUES]],
                reason="Observed distinct values.",
            )
        )
        scratchpad.value_evidence = scratchpad.value_evidence[-20:]
    elif isinstance(result, ColumnStats):
        if result.filename not in scratchpad.selected_files:
            scratchpad.selected_files.append(result.filename)
    elif isinstance(result, ColumnValueSearchResult):
        if result.filename not in scratchpad.selected_files:
            scratchpad.selected_files.append(result.filename)
    elif isinstance(result, RowSample):
        if result.filename not in scratchpad.selected_files:
            scratchpad.selected_files.append(result.filename)
    elif isinstance(result, ZipManifest):
        if result.zip_filename not in scratchpad.selected_files:
            scratchpad.selected_files.append(result.zip_filename)
    elif (
        isinstance(result, list)
        and result
        and isinstance(result[0], str)
        and scratchpad.selected_files
    ):
        scratchpad.known_sheets[scratchpad.selected_files[-1]] = cast(
            list[str], result[:MAX_VALUES]
        )

    if truncated:
        scratchpad.open_questions = (
            scratchpad.open_questions + ["More detail was available but truncated."]
        )[-8:]


def _truncate(value: str, max_length: int) -> str:
    """Truncate a string deterministically for prompt hygiene.

    Args:
        value: Candidate text value.
        max_length: Maximum retained length.

    Returns:
        str: Truncated string.
    """
    if len(value) <= max_length:
        return value
    return value[: max_length - 3] + "..."
