from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from src.predict import load_best_model, predict_one

class PredictRequest(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    return_1d: float
    return_5d: float
    return_10d: float
    ma_ratio: float
    volume_change_5d: float

class PredictResponse(BaseModel):
    model_name: str
    prediction: int
    probability: Optional[float]
    features_used: list[str]

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.artifacts = load_best_model()
    yield    

app = FastAPI(
    title = "Stock Movement ML API",
    version = "0.1.0",
    lifespan = lifespan,
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        result = predict_one(request.model_dump(), app.state.artifacts)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    

    """
    example request, send this to /predict:
    {
        "return_1d": 0.004,
        "return_5d": 0.012,
        "return_10d": 0.018,
        "ma_ratio": 1.01,
        "volume_change_5d": 0.08
    }

    example response:
    {
        "model_name": "random_forest",
        "prediction": 1,
        "probability": 0.67,
        "features_used": [
            "return_1d",
            "return_5d",
            "return_10d",
            "ma_ratio",
            "volume_change_5d"
        ]
    }

    """