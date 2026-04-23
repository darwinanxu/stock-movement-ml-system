import pandas as pd
from src.config import TICKER, START_DATE, END_DATE


def _flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [column[0] for column in df.columns]

    return df


def download_data(ticker=TICKER, start=START_DATE, end=END_DATE):
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end)
    df.reset_index(inplace=True)
    df = _flatten_yfinance_columns(df)
    return df


def save_raw_data(df):
    df.to_csv("data/raw/stock_data.csv", index=False)

if __name__ == "__main__":
    df = download_data()
    save_raw_data(df)
    print(df.head())

    
