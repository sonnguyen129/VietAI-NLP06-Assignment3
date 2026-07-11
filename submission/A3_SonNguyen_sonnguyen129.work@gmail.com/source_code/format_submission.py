"""
format_submission.py - Convert arbitrary Phase 3 predictions into Kaggle format.

Use this when your Phase 3 method does not use the self-evolution strategy runner.
Your input can be a CSV or JSON file containing test IDs and predicted values.

Examples:
    python format_submission.py --predictions my_predictions.csv --output-file submission.csv
    python format_submission.py --predictions my_predictions.json --output-file submission.csv

Accepted prediction columns/keys:
    id, predicted_value
    id, prediction
    id, answer
    id, value

Any missing, blank, null, NaN, or non-numeric prediction is written as 0.0.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


PREDICTION_KEYS = ("predicted_value", "prediction", "answer", "value")


def clean_prediction_value(value: Any, default: float = 0.0) -> float:
    """Return a finite float prediction, using default for null or invalid values."""
    if value is None:
        return default
    if isinstance(value, str):
        value = value.strip()
        if not value or value.lower() in {"null", "none", "nan"}:
            return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _load_test_ids(test_file: Path) -> list[str]:
    with test_file.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    return [str(row.get("id", idx)) for idx, row in enumerate(rows)]


def _pick_prediction(row: dict[str, Any]) -> Any:
    for key in PREDICTION_KEYS:
        if key in row:
            return row[key]
    return None


def _load_json_predictions(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "predictions" in data and isinstance(data["predictions"], list):
            data = data["predictions"]
        else:
            return {str(k): v for k, v in data.items()}

    if not isinstance(data, list):
        raise ValueError("JSON predictions must be a list, an id->value object, or contain a 'predictions' list.")

    predictions: dict[str, Any] = {}
    for row in data:
        if not isinstance(row, dict) or "id" not in row:
            continue
        predictions[str(row["id"])] = _pick_prediction(row)
    return predictions


def _load_csv_predictions(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "id" not in reader.fieldnames:
            raise ValueError("CSV predictions must include an 'id' column.")
        predictions = {}
        for row in reader:
            predictions[str(row.get("id", ""))] = _pick_prediction(row)
    return predictions


def load_predictions(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_json_predictions(path)
    if suffix == ".csv":
        return _load_csv_predictions(path)
    raise ValueError("Predictions file must be .csv or .json.")


def write_submission(predictions: dict[str, Any], test_file: Path, output_file: Path) -> int:
    test_ids = _load_test_ids(test_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "Usage", "predicted_value"])
        for test_id in test_ids:
            writer.writerow([test_id, "Public", clean_prediction_value(predictions.get(test_id))])
    return len(test_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Format arbitrary Phase 3 predictions into Kaggle submission.csv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--predictions", required=True, type=Path, help="CSV or JSON predictions file.")
    parser.add_argument("--output-file", default=Path("submission.csv"), type=Path, help="Output Kaggle CSV path.")
    parser.add_argument("--test-file", default=Path("data/test.json"), type=Path, help="Official test JSON for row order.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        predictions = load_predictions(args.predictions)
        count = write_submission(predictions, args.test_file, args.output_file)
    except Exception as exc:
        print(f"Failed to format submission: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Wrote {count} rows to {args.output_file}. Null/invalid predictions were replaced with 0.0.")


if __name__ == "__main__":
    main()
