from __future__ import annotations

import json
import os
from dataclasses import dataclass
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

@dataclass
class PredictionArtifacts:
    model: Any
    model_name: str
    feature_columns: list[str]

def load_best_model(metadata_path: str = "models/best_model.json") -> PredictionArtifacts:
    """
    Load the best trained model using a metadata file.
    Example best_model.json:
    {
        "mode_name": "logistic_regression",
        "model_path": "models/logistic_regression.joblib",
        "feature_columns": ["return_1d", "return_5d", "return_10d", "ma_ratio", "volume_change_5d"]
    }
    """

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, "r") as f:
        meta = json.load(f)

    model_path = meta["model_path"]
    model_name = meta["model_name"]
    feature_columns = meta.get("feature_columns", DEFAULT_FEATURE_COLUMNS)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = joblib.load(model_path)
    return PredictionArtifacts(
        model = model,
        model_name = model_name,
        feature_columns = feature_columns,
    )

def make_feature_frame(payload: dict, feature_columns: list[str]) -> pd.DataFrame:
    missing = [col for col in feature_columns if col not in payload]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    
    row = {col:float(payload[col]) for col in feature_columns}
    return pd.DataFrame([row], columns=feature_columns)

def predict_one(payload: dict, artifacts: PredictionArtifacts) -> dict:
    X = make_feature_frame(payload, artifacts.feature_columns)

    pred = int(artifacts.model.predict(X)[0])

    probability = None
    if hasattr(artifacts.model, "predict_proba"):
        proba = artifacts.model.predict_proba(X)[0]
        probability = float(proba[1]) #probability of positive class

    return{
        "model_name": artifacts.model_name,
        "prediction": pred,
        "probability": probability,
        "features_used": artifacts.feature_columns,
    }


