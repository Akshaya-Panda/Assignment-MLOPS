"""Inference helpers shared by the CLI and the API (no MLflow dependency).

Usage:
    python -m cardiorisk.predict --input sample_request.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from cardiorisk import config

LABELS = {0: "no_heart_disease", 1: "heart_disease"}


def load_model(path: Path = config.MODEL_PATH):
    return joblib.load(path)


def load_model_card(path: Path = config.MODEL_CARD_PATH) -> dict:
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {}


def risk_band(probability: float) -> str:
    if probability < 0.3:
        return "low"
    if probability < 0.6:
        return "moderate"
    return "high"


def to_frame(records: list[dict]) -> pd.DataFrame:
    """Order columns as in training; missing optional fields become NaN for imputation."""
    frame = pd.DataFrame.from_records(records).reindex(columns=config.FEATURE_COLUMNS)
    return frame.astype(float)


def predict_records(model, records: list[dict], threshold: float = 0.5) -> list[dict]:
    proba = model.predict_proba(to_frame(records))[:, 1]
    results = []
    for p in np.asarray(proba, dtype=float):
        label = int(p >= threshold)
        results.append(
            {
                "prediction": label,
                "label": LABELS[label],
                "probability_disease": round(p, 4),
                "confidence": round(p if label else 1 - p, 4),
                "risk_band": risk_band(p),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict heart-disease risk from JSON input")
    parser.add_argument("--input", required=True, help="JSON object or list of objects")
    parser.add_argument("--model", default=str(config.MODEL_PATH))
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text())
    records = payload if isinstance(payload, list) else [payload]
    threshold = load_model_card().get("decision_threshold", 0.5)
    print(json.dumps(predict_records(load_model(args.model), records, threshold), indent=2))


if __name__ == "__main__":
    main()
