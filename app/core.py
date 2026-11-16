
"""Core functionality for the Sports Commentary Generator."""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

REQUIRED_FIELDS = {
    "sport",
    "event_type",
    "description",
    "team",
    "player",
    "match_context",
    "commentary",
}


def load_jsonl(file_path):
    """Load records from a JSON Lines file."""
    records = []

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {file_path}, line {line_number}"
                ) from error

            records.append(record)

    return records


def validate_records(records):
    """Validate dataset structure, content, and duplicate records."""

    seen = set()

    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Every dataset record must be a JSON object.")

        if not REQUIRED_FIELDS.issubset(record.keys()):
            raise ValueError("A dataset record is missing required fields.")

        if record["sport"] not in {"football", "cricket"}:
            raise ValueError("Unsupported sport found in dataset.")

        for field in REQUIRED_FIELDS:
            if not isinstance(record[field], str) or not record[field].strip():
                raise ValueError(f"Invalid or empty field: {field}")

        unique_key = (
            record["sport"],
            record["description"],
            record["commentary"],
        )

        if unique_key in seen:
            raise ValueError("Duplicate event-commentary record found.")

        seen.add(unique_key)


def write_jsonl(file_path, records):
    """Write records into a JSON Lines file."""

    with file_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def milestone_1():
    """Load, validate, and combine football and cricket datasets."""

    football_path = DATA_DIR / "football.jsonl"
    cricket_path = DATA_DIR / "cricket.jsonl"

    if not football_path.exists():
        raise FileNotFoundError(f"Dataset not found: {football_path}")

    if not cricket_path.exists():
        raise FileNotFoundError(f"Dataset not found: {cricket_path}")

    football_records = load_jsonl(football_path)
    cricket_records = load_jsonl(cricket_path)

    # Check that each source file contains the correct sport.
    if any(record.get("sport") != "football" for record in football_records):
        raise ValueError("Football dataset contains non-football records.")

    if any(record.get("sport") != "cricket" for record in cricket_records):
        raise ValueError("Cricket dataset contains non-cricket records.")

    combined_records = football_records + cricket_records

    validate_records(combined_records)

    write_jsonl(DATA_DIR / "combined_dataset.jsonl", combined_records)

    return {
        "football": len(football_records),
        "cricket": len(cricket_records),
        "total": len(combined_records),
    }


# --- Milestone 2: Implement LoRA adapter injection and training.
def milestone_2():
    raise NotImplementedError("TODO(milestone 2)")


# --- Milestone 3: Build the Streamlit interface.
def milestone_3():
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Test commentary quality.
def milestone_4():
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Implement adapter merging for inference.
def milestone_5():
    raise NotImplementedError("TODO(milestone 5)")
