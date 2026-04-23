import pandas as pd
import pytest

from src.config import LABEL_HORIZON
from src.labels import add_labels


def test_add_labels_creates_future_return_and_label():
    df = pd.DataFrame({"Close": [100, 101, 102, 103, 104, 106]})

    result = add_labels(df)

    assert "future_return" in result.columns
    assert "label" in result.columns
    assert result.loc[0, "future_return"] == pytest.approx(0.06)
    assert result.loc[0, "label"] == 1


def test_add_labels_last_horizon_rows_have_no_future_return():
    df = pd.DataFrame({"Close": [100, 101, 102, 103, 104, 105, 106]})

    result = add_labels(df)

    assert result.tail(LABEL_HORIZON)["future_return"].isna().all()
