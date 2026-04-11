from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

DEFAULT_FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_10d",
    "ma_ratio",
    "volume_change_5d",
]

# src/predict.py -> parent is src, parent.parent is project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METADATA_PATH = PROJECT_ROOT / "models" / "best_model.json"


@dataclass
class PredictionArtifacts:
    model: Any
    model_name: str
    feature_columns: list[str]


def load_best_model(
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
) -> PredictionArtifacts:
    """
    Load the best trained model using a metadata file.

    Example best_model.json:
    {
        "model_name": "logistic_regression",
        "model_path": "models/logistic_regression.joblib",
        "feature_columns": [
            "return_1d",
            "return_5d",
            "return_10d",
            "ma_ratio",
            "volume_change_5d"
        ]
    }
    """
    metadata_path = Path(metadata_path)

    if not metadata_path.is_absolute():
        metadata_path = PROJECT_ROOT / metadata_path

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, "r") as f:
        meta = json.load(f)

    model_path = Path(meta["model_path"])
    model_name = meta["model_name"]
    feature_columns = meta.get("feature_columns", DEFAULT_FEATURE_COLUMNS)

    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)

    return PredictionArtifacts(
        model=model,
        model_name=model_name,
        feature_columns=feature_columns,
    )


def make_feature_frame(payload: dict, feature_columns: list[str]) -> pd.DataFrame:
    missing = [col for col in feature_columns if col not in payload]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    row = {col: float(payload[col]) for col in feature_columns}
    return pd.DataFrame([row], columns=feature_columns)


def predict_one(payload: dict, artifacts: PredictionArtifacts) -> dict:
    X = make_feature_frame(payload, artifacts.feature_columns)

    pred = int(artifacts.model.predict(X)[0])

    probability = None
    if hasattr(artifacts.model, "predict_proba"):
        proba = artifacts.model.predict_proba(X)[0]
        probability = float(proba[1])  # probability of positive class

    return {
        "model_name": artifacts.model_name,
        "prediction": pred,
        "probability": probability,
        "features_used": artifacts.feature_columns,
    }