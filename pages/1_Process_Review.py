"""Process dataset data access, filtering, summaries, and warning projection.

This is the UI-free extraction of the data logic from the Process review page.
It deliberately has no Streamlit imports or rendering calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESS_METADATA_FILE = PROJECT_DIR / "data" / "process_metadata.json"
ALL_FILTER = "All"
DISPLAY_COLUMNS = ("machine", "date", "time", "shift", "serial", "filename")


class ProcessFilters(TypedDict, total=False):
    machine: str
    date: str
    shift: str


class ProcessRecord(TypedDict, total=False):
    machine: str
    date: str
    time: str
    datetime: str
    shift: str
    serial: str
    source_code: str
    date_folder: str
    filename: str
    relative_path: str


def load_process_metadata(metadata_path: Path = PROCESS_METADATA_FILE) -> tuple[dict[str, Any], str | None]:
    """Load the generated process manifest and return an error instead of raising."""
    if not metadata_path.exists():
        return {}, f"Metadata file not found: {metadata_path}"
    try:
        with metadata_path.open(encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
    except OSError as error:
        return {}, f"Unable to read metadata file: {error}"
    except json.JSONDecodeError as error:
        return {}, f"Metadata is not valid JSON: {error.msg}"
    if not isinstance(metadata, dict):
        return {}, "Metadata root must be a JSON object"
    return metadata, None


def records_from(metadata: dict[str, Any]) -> list[ProcessRecord]:
    """Return only object-shaped records from the manifest."""
    records = metadata.get("records", [])
    if not isinstance(records, list):
        return []
    return [record for record in records if isinstance(record, dict)]


def unique_values(records: list[ProcessRecord], key: str) -> list[str]:
    """Build sorted filter options for a record property."""
    return sorted({str(record[key]) for record in records if record.get(key)})


def filter_options(records: list[ProcessRecord]) -> dict[str, list[str]]:
    """Return Machine, Date, and Shift options, each prefixed with the all value."""
    return {
        "machine": [ALL_FILTER, *unique_values(records, "machine")],
        "date": [ALL_FILTER, *unique_values(records, "date")],
        "shift": [ALL_FILTER, *unique_values(records, "shift")],
    }


def filter_records(records: list[ProcessRecord], filters: ProcessFilters | None = None) -> list[ProcessRecord]:
    """Apply Machine, Date, and Shift equality filters; omitted/All values pass through."""
    active_filters = filters or {}
    return [
        record
        for record in records
        if all(
            not active_filters.get(key, ALL_FILTER) or active_filters.get(key, ALL_FILTER) == ALL_FILTER
            or record.get(key) == active_filters[key]
            for key in ("machine", "date", "shift")
        )
    ]


def overview_counts(metadata: dict[str, Any], records: list[ProcessRecord] | None = None) -> dict[str, int]:
    """Return the five Process overview values with record-derived fallbacks."""
    available_records = records if records is not None else records_from(metadata)
    summary = metadata.get("summary")
    summary = summary if isinstance(summary, dict) else {}
    machine_counts = summary.get("machine_counts")
    machine_count = len(machine_counts) if isinstance(machine_counts, dict) else len(unique_values(available_records, "machine"))
    return {
        "dataset_images": int(summary.get("total_bmp_count", len(available_records))),
        "process_records": int(summary.get("record_count", len(available_records))),
        "machines": machine_count,
        "parse_errors": int(summary.get("parse_error_count", 0)),
        "data_warnings": int(summary.get("data_warning_count", 0)),
    }


def table_rows(records: list[ProcessRecord]) -> list[dict[str, Any]]:
    """Project process records to the fields required by the Process table."""
    return [{key: record.get(key, "") for key in DISPLAY_COLUMNS} for record in records]


def warning_rows(metadata: dict[str, Any]) -> list[dict[str, str]]:
    """Normalize data-quality warnings for a consumer table or API response."""
    warnings = metadata.get("data_warnings", [])
    if not isinstance(warnings, list):
        return []
    return [
        {
            "relative_path": str(warning.get("relative_path", "")),
            "warnings": "; ".join(str(item) for item in warning.get("warnings", [])),
        }
        for warning in warnings
        if isinstance(warning, dict)
    ]
