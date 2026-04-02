import os
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression

from src.config import TRAIN_SPLIT_DATE
from src.data_loader import download_data
from src.features import add_features
from src.labels import add_labels
from src.evaluate import evaluate_classifier


FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_10d",
    "ma_ratio",
    "volume_change_5d",
]


def prepare_dataset():
    df = download_data()
    df = add_features(df)
    df = add_labels(df)

    df = df.dropna().copy()
    
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/model_dataset.csv", index=False)


    train_df = df[df["Date"] < TRAIN_SPLIT_DATE].copy()
    test_df = df[df["Date"] >= TRAIN_SPLIT_DATE].copy()

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["label"]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["label"]

    return X_train, y_train, X_test, y_test, train_df, test_df


def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def save_model(model, path="models/best_model.joblib"):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, path)


if __name__ == "__main__":
    X_train, y_train, X_test, y_test, train_df, test_df = prepare_dataset()

    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print("Feature columns:", FEATURE_COLUMNS)

    model = train_model(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = evaluate_classifier(y_test, y_pred, y_prob)

    print("\nTest metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    save_model(model)
    print("\nSaved model to models/best_model.joblib")