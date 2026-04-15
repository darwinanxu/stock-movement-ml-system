# Stock Movement ML System

An end-to-end machine learning system for predicting short-term stock movement, with model serving via FastAPI and Docker.

---

## Overview

This project includes:

* Data ingestion from Yahoo Finance (SPY)
* Feature engineering on price and volume signals
* Model training and selection
* Model serialization and metadata tracking
* REST API for prediction using FastAPI
* Dockerized deployment

---

## Problem Definition

* Task: Binary classification
* Goal: Predict whether SPY will increase by more than 1% over the next 5 days

Label:

```
future_5d_return > 1%
```

---

## Features

* return_1d
* return_5d
* return_10d
* ma_ratio
* volume_change_5d

---

## Project Structure

```
stock-movement-ml-system/
├── app/
│   └── main.py
├── src/
│   ├── train.py
│   ├── train_lr.py
│   ├── train_rf.py
│   ├── train_torch.py
│   └── predict.py
├── models/
│   ├── *.joblib
│   └── best_model.json
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Models

* Logistic Regression
* Random Forest
* PyTorch MLP

Model selection is based on F1 score.
The best model is saved along with metadata in:

```
models/best_model.json
```

Example:

```json
{
  "model_name": "random_forest",
  "model_path": "models/random_forest.joblib",
  "feature_columns": [
    "return_1d",
    "return_5d",
    "return_10d",
    "ma_ratio",
    "volume_change_5d"
  ]
}
```

---

## Training

Run:

```
python src/train.py
```

This will:

* Train all models
* Evaluate performance
* Save the best model
* Write metadata to best_model.json

---

## API

Start locally:

```
uvicorn app.main:app --reload
```

Docs:

```
http://localhost:8000/docs
```

### Endpoints

**GET /health**

```
{"status": "ok"}
```

---

**POST /predict**

Request:

```json
{
  "return_1d": 0.004,
  "return_5d": 0.012,
  "return_10d": 0.018,
  "ma_ratio": 1.01,
  "volume_change_5d": 0.08
}
```

Response:

```json
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
```

---

## Docker

Build:

```
docker build -t stock-ml-api .
```

Run:

```
docker run -p 8000:8000 stock-ml-api
```

Then open:

```
http://localhost:8000/docs
```

---

## Notes

* Model paths are relative for Docker compatibility
* Models are included in the image for reproducible serving
* API loads model once at startup

---

## Future Work

* Add tests
* Add experiment tracking
* Add CI/CD
* Improve features
* Deploy to cloud
