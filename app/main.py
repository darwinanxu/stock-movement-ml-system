from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from src.predict import load_best_model, predict_one


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    title="Stock Movement ML API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        result = predict_one(request.model_dump(), app.state.artifacts)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {e}")