import pandas as pd
from src.config import LABEL_HORIZON, RETURN_THRESHOLD

def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["future_return"] = df["Close"].shift(-LABEL_HORIZON)/df["Close"] - 1
    df["label"] = (df["future_return"] > RETURN_THRESHOLD).astype(int)

    return df