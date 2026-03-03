"""Public interface for capsule inspection and data-loading tools."""

from science_bot.pipeline.resolution.tools.capsule import (
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
from science_bot.pipeline.resolution.tools.schemas import (
    CapsuleManifest,
    ColumnInfo,
    ColumnSearchResult,
    ColumnStats,
    ColumnValues,
    ColumnValueSearchResult,
    FileInfo,
    FileSchema,
    RowSample,
    ZipEntry,
    ZipManifest,
)

AVAILABLE_TOOLS_TEXT = """
Available tools:
- list_zip_contents(zip_filename)
- list_excel_sheets(filename)
- find_files_with_column(query)
- get_file_schema(filename)
- search_columns(filename, query)
- get_column_values(filename, column)
- get_column_stats(filename, column)
- search_column_for_value(filename, column, query)
- get_row_sample(filename, columns, n=10, random_sample=False)
""".strip()

__all__ = [
    "AVAILABLE_TOOLS_TEXT",
    # Discovery
    "list_capsule_files",
    "list_zip_contents",
    "list_excel_sheets",
    "find_files_with_column",
    # Schema inspection
    "get_file_schema",
    "search_columns",
    # Value inspection
    "get_column_values",
    "get_column_stats",
    "search_column_for_value",
    # Row preview
    "get_row_sample",
    # Data assembly
    "load_dataframe",
    # Schemas
    "CapsuleManifest",
    "ColumnInfo",
    "ColumnSearchResult",
    "ColumnStats",
    "ColumnValueSearchResult",
    "ColumnValues",
    "FileInfo",
    "FileSchema",
    "RowSample",
    "ZipEntry",
    "ZipManifest",
]
