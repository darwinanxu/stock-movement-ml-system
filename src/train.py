from src.data_loader import download_data
from src.features import add_features
from src.labels import add_labels

if __name__  == "__main__":
    df = download_data()
    df = add_features(df)
    df = add_labels(df)

    print(df[["return_5d", "ma5", "future_return","label"]].head(20))