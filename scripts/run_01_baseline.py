"""submit-01 baseline — CatBoost defaults + GroupKFold(Race+Year), 5 folds.

Trains 5 CatBoost models with strict defaults (no class_weights, no clipping, no FE).
Predicts OOF on train + averages 5 fold-models on test. Writes submissions/01_baseline.csv.

Output:
    - per-fold AUC / Brier / logloss / n_val / frac_2023_in_val
    - OOF aggregate AUC / Brier / logloss
    - sanity checks (n_rows, prob range, NaN)
    - submissions/01_baseline.csv (id, PitNextLap)

Run:
    .venv/Scripts/python.exe scripts/run_01_baseline.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data import load_test, load_train, make_group_kfold  # noqa: E402
from src.evaluate import compute_metrics  # noqa: E402
from src.submit import write_submission  # noqa: E402

SEED = 42
N_SPLITS = 5
CAT_FEATURES = ["Driver", "Compound", "Race", "Year"]
TARGET = "PitNextLap"
ID_COL = "id"


def nan_audit(df: pd.DataFrame, split_name: str) -> None:
    """Raise if any column exceeds 99% NaN — ALICE lesson, mandatory pre-flight."""
    for col in df.columns:
        nan_pct = df[col].isna().mean()
        if nan_pct > 0.99:
            raise ValueError(f"{col} is {nan_pct:.0%} NaN on {split_name}")
    total_nan = df.isna().sum().sum()
    print(f"  [{split_name}] {len(df):>7} rows x {len(df.columns)} cols | total NaN = {total_nan}")


def prepare(df: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
    """Drop id + target; cast cat features to string for CatBoost.

    Year (int64) must be string per CatBoost cat-feature contract:
    'Categorical feature values must have the type string'.
    """
    cols_to_drop = [ID_COL] + ([TARGET] if is_train else [])
    X = df.drop(columns=cols_to_drop).copy()
    for c in CAT_FEATURES:
        X[c] = X[c].astype(str)
    return X


def main() -> None:
    np.random.seed(SEED)

    print("=" * 78)
    print("submit-01 baseline — CatBoost defaults + GroupKFold(Race+Year), 5 folds")
    print("=" * 78)

    print("\n[1/5] Load data + NaN audit")
    train = load_train()
    test = load_test()
    nan_audit(train, "train")
    nan_audit(test, "test")

    X = prepare(train, is_train=True)
    y = train[TARGET].astype(int).values
    X_test = prepare(test, is_train=False)
    test_ids = test[ID_COL]

    assert list(X.columns) == list(X_test.columns), (
        f"train cols {list(X.columns)} != test cols {list(X_test.columns)}"
    )
    print(f"  features ({len(X.columns)}): {list(X.columns)}")
    print(f"  cat_features ({len(CAT_FEATURES)}): {CAT_FEATURES}")
    print(f"  target rate train = {y.mean():.4f} (n_pos = {int(y.sum())})")

    print("\n[2/5] Train 5 folds")
    oof = np.zeros(len(X), dtype=np.float64)
    test_preds_per_fold = np.zeros((N_SPLITS, len(X_test)), dtype=np.float64)
    fold_metrics: list[dict[str, float | int]] = []

    splits = list(make_group_kfold(train, n_splits=N_SPLITS))
    for fold_idx, (tr_idx, va_idx) in enumerate(splits):
        t0 = time.time()
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        model = CatBoostClassifier(
            iterations=1000,
            random_seed=SEED,
            eval_metric="AUC",
            early_stopping_rounds=50,
            cat_features=CAT_FEATURES,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va))

        va_pred = model.predict_proba(X_va)[:, 1]
        oof[va_idx] = va_pred
        test_preds_per_fold[fold_idx] = model.predict_proba(X_test)[:, 1]

        m = compute_metrics(y_va, va_pred)
        frac_2023 = float((train.iloc[va_idx]["Year"] == 2023).mean())
        n_val = int(len(va_idx))
        best_iter = int(model.get_best_iteration())
        fold_metrics.append(
            {
                "fold": fold_idx,
                "n_val": n_val,
                "frac_2023": frac_2023,
                "best_iter": best_iter,
                **m,
            }
        )
        print(
            f"  fold {fold_idx} | n_val={n_val:>6} | frac_2023={frac_2023:.3f} | "
            f"best_iter={best_iter:>4} | AUC={m['auc']:.4f} | "
            f"Brier={m['brier']:.4f} | logloss={m['logloss']:.4f} | "
            f"{time.time() - t0:.1f}s"
        )

    print("\n[3/5] OOF aggregate")
    oof_metrics = compute_metrics(y, oof)
    auc_per_fold = np.array([fm["auc"] for fm in fold_metrics])
    print(f"  OOF AUC = {oof_metrics['auc']:.4f}")
    print(f"  OOF Brier = {oof_metrics['brier']:.4f}")
    print(f"  OOF logloss = {oof_metrics['logloss']:.4f}")
    print(f"  per-fold AUC mean = {auc_per_fold.mean():.4f} +/- {auc_per_fold.std():.4f}")
    print(f"  per-fold AUC range = [{auc_per_fold.min():.4f}, {auc_per_fold.max():.4f}]")

    print("\n[4/5] Test prediction = mean of 5 fold-models")
    test_pred = test_preds_per_fold.mean(axis=0)
    print(f"  test_pred shape    = {test_pred.shape}")
    print(f"  test_pred range    = [{test_pred.min():.4f}, {test_pred.max():.4f}]")
    print(f"  test_pred mean     = {test_pred.mean():.4f}")
    print(f"  test_pred median   = {np.median(test_pred):.4f}")
    assert test_pred.shape == (188_165,), f"expected 188165 test rows, got {test_pred.shape}"
    assert (test_pred >= 0).all() and (test_pred <= 1).all(), "probs out of [0,1]"
    assert not np.isnan(test_pred).any(), "NaN in test_pred"

    print("\n[5/5] Write submission")
    path = write_submission(
        test_ids=test_ids,
        y_pred=pd.Series(test_pred, name=TARGET),
        step_num=1,
        approach_tag="baseline",
    )
    head = pd.read_csv(path, nrows=5)
    print(f"  CSV head:\n{head.to_string(index=False)}")

    print("\n" + "=" * 78)
    print("DONE — submit-01 baseline")
    print(
        f"OOF AUC = {oof_metrics['auc']:.4f} | per-fold AUC mean = "
        f"{auc_per_fold.mean():.4f} +/- {auc_per_fold.std():.4f}"
    )
    print(f"CSV     = {path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
