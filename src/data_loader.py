import yfinance as yf
import pandas as pd
from src.config import TICKER, START_DATE, END_DATE

def download_data():
    df = yf.download(TICKER, start=START_DATE, end=END_DATE)
    df.reset_index(inplace=True)
    return df

def save_raw_data(df):
    df.to_csv("data/raw/stock_data.csv", index=False)

if __name__ == "__main__":
    df = download_data()
    save_raw_data(df)
    print(df.head())

    