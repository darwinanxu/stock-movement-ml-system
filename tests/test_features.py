import pandas as pd
import pytest

from src.features import add_features


def test_add_features_creates_expected_columns():
    df = pd.DataFrame(
        {
            "Close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            "Volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000],
        }
    )

    result = add_features(df)

    expected_columns = {
        "return_1d",
        "return_5d",
        "return_10d",
        "ma5",
        "ma10",
        "ma_ratio",
        "volume_change_5d",
    }
    assert expected_columns.issubset(result.columns)


def test_add_features_calculates_returns_and_moving_average_ratio():
    df = pd.DataFrame(
        {
            "Close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            "Volume": [1000, 1000, 1000, 1000, 1000, 1500, 1500, 1500, 1500, 1500],
        }
    )

    result = add_features(df)

    assert result.loc[1, "return_1d"] == pytest.approx(0.01)
    assert result.loc[5, "return_5d"] == pytest.approx(0.05)
    assert result.loc[5, "volume_change_5d"] == pytest.approx(0.5)
    assert result.loc[9, "ma_ratio"] == pytest.approx(
        result.loc[9, "ma5"] / result.loc[9, "ma10"]
    )
