"""Tests for src/data.py - data loading + group-aware CV."""

from __future__ import annotations

import pandas as pd


def test_load_train_returns_dataframe_with_expected_columns():
    from src.data import load_train

    df = load_train()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 439140
    assert "PitNextLap" in df.columns
    assert df["PitNextLap"].nunique() == 2


def test_load_test_returns_dataframe_no_target():
    from src.data import load_test

    df = load_test()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 188165
    assert "PitNextLap" not in df.columns


def test_make_group_kfold_splits_by_race_year():
    from src.data import load_train, make_group_kfold

    df = load_train()
    splits = list(make_group_kfold(df, n_splits=5))
    assert len(splits) == 5
    for train_idx, val_idx in splits:
        train_groups = set(
            zip(df.iloc[train_idx]["Race"], df.iloc[train_idx]["Year"], strict=False)
        )
        val_groups = set(zip(df.iloc[val_idx]["Race"], df.iloc[val_idx]["Year"], strict=False))
        assert train_groups.isdisjoint(val_groups), "Race+Year leak between folds"
