from __future__ import annotations

import csv
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


EVALUATION_FIELD = "evidence"

ROUND_DIGITS = 3 # Number of decimal places to round metrics to in the output CSV files.

CATEGORIES = (
    "condition",
    "activities",
    "execution_units",
    "execution_sites",
    "access_requirements",
    "booking_methods",
    "booking_actors",
    "expected_access_time",
)


@dataclass
class Counts:
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    exact_matches: int = 0
    compared_rows: int = 0

    def add(self, other: "Counts") -> None:
        self.tp += other.tp
        self.tn += other.tn
        self.fp += other.fp
        self.fn += other.fn
        self.exact_matches += other.exact_matches
        self.compared_rows += other.compared_rows


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read JSON file '{path}': {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"JSON root in '{path}' must be an object.")
    return data


def normalise_text(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)

    text = unicodedata.normalize("NFKC", str(value))
    text = text.casefold()
    text = text.replace("’", "'").replace("`", "'")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*([/,+;:()\-])\s*", r"\1", text)
    return text or None


def _extract_field(category_value: Any, field: str) -> Any:
    if not isinstance(category_value, dict):
        return category_value

    if field in category_value:
        return category_value[field]

    # Convenience fallback between singular and plural interpreted fields.
    if field == "interpreted_values" and "interpreted_value" in category_value:
        return category_value["interpreted_value"]
    if field == "interpreted_value" and "interpreted_values" in category_value:
        return category_value["interpreted_values"]

    return None


def normalised_values(record: dict[str, Any], category: str, field: str) -> tuple[str, ...]:
    raw = _extract_field(record.get(category), field)

    if raw is None:
        return ()

    if isinstance(raw, list):
        values = raw
    else:
        values = [raw]

    normalised = {
        item
        for item in (normalise_text(value) for value in values)
        if item is not None
    }
    return tuple(sorted(normalised))


def index_records(data: dict[str, Any], source_name: str) -> dict[int, dict[str, Any]]:
    records = data.get("records")
    if not isinstance(records, list):
        raise ValueError(f"'{source_name}' does not contain a valid records array.")

    indexed: dict[int, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"'{source_name}' contains a non-object record.")

        row = record.get("table_row")
        if not isinstance(row, int) or row < 1:
            raise ValueError(f"'{source_name}' contains an invalid table_row: {row!r}.")
        if row in indexed:
            raise ValueError(f"'{source_name}' contains duplicate table_row {row}.")
        indexed[row] = record

    return indexed


def compare_pair(gold_values: tuple[str, ...], predicted_values: tuple[str, ...]) -> Counts:
    gold_present = bool(gold_values)
    predicted_present = bool(predicted_values)
    exact = gold_values == predicted_values

    result = Counts(compared_rows=1, exact_matches=int(exact))

    if not gold_present and not predicted_present:
        result.tn = 1
    elif exact:
        result.tp = 1
    elif gold_present and predicted_present:
        # A wrong non-empty extraction is simultaneously a missed gold value
        # and an unsupported predicted value.
        result.fp = 1
        result.fn = 1
    elif predicted_present:
        result.fp = 1
    else:
        result.fn = 1

    return result


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def metrics(counts: Counts) -> dict[str, float | int]:
    accuracy = safe_divide(counts.exact_matches, counts.compared_rows)
    precision = safe_divide(counts.tp, counts.tp + counts.fp)
    recall = safe_divide(counts.tp, counts.tp + counts.fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    return {
        "accuracy": round(accuracy, ROUND_DIGITS),
        "precision": round(precision, ROUND_DIGITS),
        "recall": round(recall, ROUND_DIGITS),
        "f1_score": round(f1, ROUND_DIGITS),
        "tp": counts.tp,
        "tn": counts.tn,
        "fp": counts.fp,
        "fn": counts.fn,
        "exact_matches": counts.exact_matches,
        "compared_rows": counts.compared_rows,
    }


def infer_model_name(prediction_path: Path, prediction_data: dict[str, Any]) -> str:
    candidate_paths: Iterable[tuple[str, ...]] = (
        ("model",),
        ("model_name",),
        ("metadata", "model"),
        ("metadata", "model_name"),
        ("llm", "model"),
        ("config", "model"),
    )

    for path in candidate_paths:
        current: Any = prediction_data
        for key in path:
            if not isinstance(current, dict) or key not in current:
                current = None
                break
            current = current[key]
        if isinstance(current, str) and current.strip():
            return current.strip()

    stem = prediction_path.stem
    stem = re.sub(r"_data$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_\d{8}T\d{6}Z$", "", stem)
    parts = stem.split("_")
    if len(parts) >= 3:
        return "_".join(parts[2:])
    return stem


def evaluate(
    gold_standard_path: Path,
    prediction_path: Path,
    output_dir: Path,
    field: str = EVALUATION_FIELD,
    model_name: str | None = None,
) -> tuple[Path, Path, dict[str, float | int]]:
    gold_data = load_json(gold_standard_path)
    prediction_data = load_json(prediction_path)

    gold_records = index_records(gold_data, str(gold_standard_path))
    predicted_records = index_records(prediction_data, str(prediction_path))

    all_rows = sorted(set(gold_records) | set(predicted_records))
    if not all_rows:
        raise ValueError("No records are available for evaluation.")

    per_category: list[dict[str, Any]] = []
    aggregate_counts = Counts()

    for category in CATEGORIES:
        category_counts = Counts()

        for row in all_rows:
            gold_record = gold_records.get(row, {})
            predicted_record = predicted_records.get(row, {})
            gold_values = normalised_values(gold_record, category, field)
            predicted_values = normalised_values(predicted_record, category, field)
            category_counts.add(compare_pair(gold_values, predicted_values))

        category_metrics = metrics(category_counts)
        per_category.append({"category": category, **category_metrics})
        aggregate_counts.add(category_counts)

    aggregate_metrics = metrics(aggregate_counts)
    resolved_model_name = model_name or infer_model_name(prediction_path, prediction_data)

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", resolved_model_name).strip("_") or "model"

    details_path = output_dir / f"evaluation_details_{safe_model}.csv"
    details_fields: list[str] = [
        "category",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "tp",
        "tn",
        "fp",
        "fn",
        "exact_matches",
        "compared_rows",
    ]
    with details_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=details_fields)
        writer.writeheader()
        writer.writerows(per_category)

    summary_path = output_dir / "evaluation_summary.csv"
    summary_fields: list[str] = [
        "model",
        "evaluation_field",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "tp",
        "tn",
        "fp",
        "fn",
        "exact_matches",
        "compared_rows",
        "gold_standard",
        "prediction",
    ]
    write_header = not summary_path.exists() or summary_path.stat().st_size == 0
    with summary_path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "model": resolved_model_name,
                "evaluation_field": field,
                **aggregate_metrics,
                "gold_standard": str(gold_standard_path),
                "prediction": str(prediction_path),
            }
        )

    return details_path, summary_path, aggregate_metrics
