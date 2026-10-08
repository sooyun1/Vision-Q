"""Build an app-ready index for the X-ray process image dataset.

Run from the project root:
    python fixed_project/build_process_metadata.py
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_DIR / "dataset" / "test1" / "yolov3" / "X선이물검출기(06.23_09.22)"
OUTPUT_PATH = PROJECT_DIR / "data" / "process_metadata.json"
MACHINE_PATTERN = re.compile(r"^(?P<machine>\d+호기)")
DATE_FOLDER_PATTERN = re.compile(r"^(?P<serial>.+?)_(?P<date>\d{8})_NgImage$")
FILENAME_PATTERN = re.compile(
    r"^(?:\d+_)?(?P<date>\d{8})_(?P<time>\d{6})\((?P<source_code>[^)]+)\)\.bmp$",
    re.IGNORECASE,
)
SHIFT_RULE = {
    "day_shift": "06:00-17:59",
    "night_shift": "18:00-05:59",
}


def shift_for(hour: int) -> str:
    return "주간조" if 6 <= hour <= 17 else "야간조"


def parse_image(image_path: Path, machine_dir: Path) -> tuple[dict[str, Any], list[str], list[str]]:
    """Return one record, parsing errors, and non-blocking data consistency warnings."""
    errors: list[str] = []
    warnings: list[str] = []
    folder_match = DATE_FOLDER_PATTERN.match(image_path.parent.name)
    name_match = FILENAME_PATTERN.match(image_path.name)
    machine_match = MACHINE_PATTERN.match(machine_dir.name)

    if not machine_match:
        errors.append("machine directory name does not start with '<number>호기'")
    if not folder_match:
        errors.append("parent directory does not match '<serial>_YYYYMMDD_NgImage'")
    if not name_match:
        errors.append("filename does not match '*_YYYYMMDD_HHMMSS(code).bmp'")

    date_value = name_match.group("date") if name_match else None
    time_value = name_match.group("time") if name_match else None
    datetime_value = None
    shift = None
    if date_value and time_value:
        try:
            parsed = datetime.strptime(f"{date_value}{time_value}", "%Y%m%d%H%M%S")
            datetime_value = parsed.isoformat(timespec="seconds")
            shift = shift_for(parsed.hour)
        except ValueError:
            errors.append("filename timestamp is not a valid calendar date/time")

    if folder_match and date_value and folder_match.group("date") != date_value:
        warnings.append("date folder and filename date do not match")

    record = {
        "machine": machine_match.group("machine") if machine_match else machine_dir.name,
        "date": date_value,
        "time": time_value,
        "datetime": datetime_value,
        "shift": shift,
        "serial": folder_match.group("serial") if folder_match else None,
        "source_code": name_match.group("source_code") if name_match else None,
        "date_folder": image_path.parent.name,
        "filename": image_path.name,
        "relative_path": image_path.relative_to(PROJECT_DIR).as_posix(),
    }
    return record, errors, warnings


def nested_counts(records: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, int]]]:
    counts: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(Counter))
    for record in records:
        if record["date"] and record["shift"]:
            counts[record["machine"]][record["date"]][record["shift"]] += 1
    return {
        machine: {date: dict(sorted(shifts.items())) for date, shifts in sorted(dates.items())}
        for machine, dates in sorted(counts.items())
    }


def build_metadata() -> dict[str, Any]:
    if not SOURCE_ROOT.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {SOURCE_ROOT}")

    machine_dirs = sorted(
        (path for path in SOURCE_ROOT.iterdir() if path.is_dir() and MACHINE_PATTERN.match(path.name)),
        key=lambda path: path.name,
    )
    records: list[dict[str, Any]] = []
    parse_errors: list[dict[str, Any]] = []
    data_warnings: list[dict[str, Any]] = []
    for machine_dir in machine_dirs:
        for image_path in sorted(machine_dir.rglob("*.bmp"), key=lambda path: path.as_posix().casefold()):
            record, errors, warnings = parse_image(image_path, machine_dir)
            records.append(record)
            if errors:
                parse_errors.append({"relative_path": record["relative_path"], "errors": errors})
            if warnings:
                data_warnings.append({"relative_path": record["relative_path"], "warnings": warnings})

    machine_counts = Counter(record["machine"] for record in records)
    date_counts = Counter(record["date"] for record in records if record["date"])
    shift_counts = Counter(record["shift"] for record in records if record["shift"])
    return {
        "source": {
            "relative_root": SOURCE_ROOT.relative_to(PROJECT_DIR).as_posix(),
            "machine_directories": [path.name for path in machine_dirs],
        },
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "shift_rule": SHIFT_RULE,
        "summary": {
            "total_bmp_count": len(records),
            "record_count": len(records),
            "machine_counts": dict(sorted(machine_counts.items())),
            "date_counts": dict(sorted(date_counts.items())),
            "shift_counts": dict(sorted(shift_counts.items())),
            "machine_date_shift_counts": nested_counts(records),
            "parse_error_count": len(parse_errors),
            "data_warning_count": len(data_warnings),
        },
        "parse_errors": parse_errors,
        "data_warnings": data_warnings,
        "records": records,
    }


def main() -> None:
    metadata = build_metadata()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = metadata["summary"]
    print(f"Wrote {OUTPUT_PATH}")
    print(f"BMP files: {summary['total_bmp_count']}")
    print(f"By machine: {summary['machine_counts']}")
    print(f"Parse errors: {summary['parse_error_count']}")


if __name__ == "__main__":
    main()
