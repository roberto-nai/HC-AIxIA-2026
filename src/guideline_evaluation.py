from __future__ import annotations

import csv
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read JSON file '{path}': {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"JSON root in '{path}' must be an object.")
    return data


def _normalise(value: str, *, case_sensitive: bool, trim_whitespace: bool) -> str:
    text = unicodedata.normalize("NFKC", value)
    text = text.replace("’", "'").replace("`", "'")
    if trim_whitespace:
        text = re.sub(r"\s+", " ", text).strip()
    if not case_sensitive:
        text = text.casefold()
    return text


def _prediction_names(data: dict[str, Any]) -> list[str]:
    guidelines = data.get("guidelines")
    if not isinstance(guidelines, list):
        raise ValueError("Prediction JSON does not contain a valid 'guidelines' array.")

    names: list[str] = []
    for item in guidelines:
        if isinstance(item, str):
            name = item
        elif isinstance(item, dict):
            name = item.get("name")
        else:
            name = None
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Each predicted guideline must contain a non-empty name.")
        names.append(name)
    return names


def _infer_model_name(prediction_path: Path, data: dict[str, Any]) -> str:
    candidates: Iterable[tuple[str, ...]] = (
        ("model",),
        ("model_name",),
        ("metadata", "model"),
        ("llm_execution", "model"),
    )
    for candidate in candidates:
        current: Any = data
        for key in candidate:
            if not isinstance(current, dict) or key not in current:
                current = None
                break
            current = current[key]
        if isinstance(current, str) and current.strip():
            return current.strip()

    stem = re.sub(r"_(data|metadata)$", "", prediction_path.stem, flags=re.IGNORECASE)
    return stem


def evaluate_guidelines(
    *,
    gold_standard_path: Path,
    prediction_path: Path,
    output_dir: Path,
    model_name: str | None = None,
) -> tuple[Path, Path, dict[str, float | int]]:
    gold_data = _load_json(gold_standard_path)
    prediction_data = _load_json(prediction_path)

    policy = gold_data.get("matching_policy", {})
    if not isinstance(policy, dict):
        raise ValueError("Gold-standard matching_policy must be an object.")
    case_sensitive = bool(policy.get("case_sensitive", False))
    trim_whitespace = bool(policy.get("trim_whitespace", True))
    use_aliases = bool(policy.get("use_accepted_matches", True))

    gold_items = gold_data.get("gold_standard")
    if not isinstance(gold_items, list) or not gold_items:
        raise ValueError("Gold-standard JSON does not contain a valid gold_standard array.")

    alias_to_gold: dict[str, str] = {}
    canonical_names: list[str] = []
    for item in gold_items:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError("Each gold-standard item must contain a name.")
        canonical = item["name"].strip()
        canonical_names.append(canonical)
        aliases = item.get("accepted_matches", []) if use_aliases else []
        candidates = [canonical, *aliases]
        for candidate in candidates:
            if not isinstance(candidate, str) or not candidate.strip():
                continue
            key = _normalise(
                candidate,
                case_sensitive=case_sensitive,
                trim_whitespace=trim_whitespace,
            )
            previous = alias_to_gold.get(key)
            if previous is not None and previous != canonical:
                raise ValueError(
                    f"Ambiguous accepted match '{candidate}' maps to both "
                    f"'{previous}' and '{canonical}'."
                )
            alias_to_gold[key] = canonical

    predicted_names = _prediction_names(prediction_data)
    matched_gold: set[str] = set()
    unsupported_predictions: list[str] = []
    duplicate_predictions: list[str] = []
    rows: list[dict[str, str]] = []

    for predicted in predicted_names:
        key = _normalise(
            predicted,
            case_sensitive=case_sensitive,
            trim_whitespace=trim_whitespace,
        )
        canonical = alias_to_gold.get(key)
        if canonical is None:
            unsupported_predictions.append(predicted)
            rows.append({"status": "FP", "gold_name": "", "predicted_name": predicted})
        elif canonical in matched_gold:
            duplicate_predictions.append(predicted)
            rows.append(
                {"status": "DUPLICATE", "gold_name": canonical, "predicted_name": predicted}
            )
        else:
            matched_gold.add(canonical)
            rows.append({"status": "TP", "gold_name": canonical, "predicted_name": predicted})

    missing_gold = [name for name in canonical_names if name not in matched_gold]
    for missing in missing_gold:
        rows.append({"status": "FN", "gold_name": missing, "predicted_name": ""})

    tp = len(matched_gold)
    fp = len(unsupported_predictions)
    fn = len(missing_gold)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    # Entity-level accuracy for an open extraction set (Jaccard similarity).
    accuracy = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    exact_set_match = int(fp == 0 and fn == 0)

    metrics: dict[str, float | int] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gold_count": len(canonical_names),
        "prediction_count": len(predicted_names),
        "duplicate_predictions": len(duplicate_predictions),
        "exact_set_match": exact_set_match,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_model = model_name or _infer_model_name(prediction_path, prediction_data)
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", resolved_model).strip("_") or "model"

    details_path = output_dir / f"guideline_evaluation_details_{safe_model}.csv"
    with details_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("status", "gold_name", "predicted_name"),
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_dir / "guideline_evaluation_summary.csv"
    fields = (
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "tp",
        "fp",
        "fn",
        "gold_count",
        "prediction_count",
        "duplicate_predictions",
        "exact_set_match",
        "gold_standard",
        "prediction",
    )
    write_header = not summary_path.exists() or summary_path.stat().st_size == 0
    with summary_path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "model": resolved_model,
                **metrics,
                "gold_standard": str(gold_standard_path),
                "prediction": str(prediction_path),
            }
        )

    return details_path, summary_path, metrics
