"""Data loading + group-aware CV for f1-pitstops-s6e5."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = REPO_ROOT / "data" / "raw"


def load_train() -> pd.DataFrame:
    """Load train.csv from data/raw/. Assumes kaggle download already ran."""
    path = DATA_RAW / "train.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: "
            f"kaggle competitions download -c playground-series-s6e5 -p {DATA_RAW}"
        )
    return pd.read_csv(path, engine="c")


def load_test() -> pd.DataFrame:
    """Load test.csv from data/raw/."""
    path = DATA_RAW / "test.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: "
            f"kaggle competitions download -c playground-series-s6e5 -p {DATA_RAW}"
        )
    return pd.read_csv(path, engine="c")


def make_group_kfold(
    df: pd.DataFrame, n_splits: int = 5
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """GroupKFold by Race+Year - prevents intra-race leak across folds."""
    groups = df["Race"].astype(str) + "-" + df["Year"].astype(str)
    gkf = GroupKFold(n_splits=n_splits)
    yield from gkf.split(df, df["PitNextLap"], groups=groups)
