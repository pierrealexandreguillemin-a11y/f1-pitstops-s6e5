"""submit-01 baseline kernel — CatBoost defaults + GroupKFold(Race+Year), 5 folds.

Self-contained Kaggle kernel (inlines minimal src.* helpers so the kernel only
needs the competition data attached; no kernel_sources / dataset_sources).

Repo traceability:
    repo:    https://github.com/-/f1-pitstops-s6e5  (local-only at push time)
    git_ref: f1d4a7e
    version: v1
    written: 2026-05-28

Output (in /kaggle/working/):
    01_baseline.csv      submission file (id, PitNextLap)
    fold_metrics.csv     per-fold AUC / Brier / logloss / frac_2023 / best_iter
    oof_summary.json     OOF aggregate + sanity checks

Run on Kaggle (CPU, ~5-15 min, RAM <16 GB):
    Auto. The kernel reads /kaggle/input/playground-series-s6e5/{train,test}.csv
    and writes outputs to /kaggle/working/.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

# ---------- Configuration ----------
INPUT = Path("/kaggle/input/playground-series-s6e5")
OUTPUT = Path("/kaggle/working")
SEED = 42
N_SPLITS = 5
CAT_FEATURES = ["Driver", "Compound", "Race", "Year"]
TARGET = "PitNextLap"
ID_COL = "id"
EXPECTED_TEST_ROWS = 188_165


# ---------- Inline helpers (mirror src.* from repo) ----------
def load_train() -> pd.DataFrame:
    return pd.read_csv(INPUT / "train.csv", engine="c")


def load_test() -> pd.DataFrame:
    return pd.read_csv(INPUT / "test.csv", engine="c")


def make_group_kfold(df: pd.DataFrame, n_splits: int = N_SPLITS):
    """GroupKFold by Race + Year. Prevents intra-race leak across folds."""
    groups = df["Race"].astype(str) + "-" + df["Year"].astype(str)
    gkf = GroupKFold(n_splits=n_splits)
    yield from gkf.split(df, df[TARGET], groups=groups)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "auc": float(roc_auc_score(y_true, y_pred)),
        "logloss": float(log_loss(y_true, np.clip(y_pred, 1e-7, 1 - 1e-7))),
        "brier": float(brier_score_loss(y_true, y_pred)),
    }


def nan_audit(df: pd.DataFrame, split_name: str) -> None:
    for col in df.columns:
        nan_pct = df[col].isna().mean()
        if nan_pct > 0.99:
            raise ValueError(f"{col} is {nan_pct:.0%} NaN on {split_name}")
    total_nan = int(df.isna().sum().sum())
    print(f"  [{split_name}] {len(df):>7} rows x {len(df.columns)} cols | total NaN = {total_nan}")


def prepare(df: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
    """Drop id (+ target); cast cat features to string per CatBoost contract."""
    cols_to_drop = [ID_COL] + ([TARGET] if is_train else [])
    X = df.drop(columns=cols_to_drop).copy()
    for c in CAT_FEATURES:
        X[c] = X[c].astype(str)
    return X


# ---------- Main pipeline ----------
def main() -> None:
    np.random.seed(SEED)
    t_start = time.time()

    print("=" * 78)
    print("submit-01 baseline - CatBoost defaults + GroupKFold(Race+Year), 5 folds")
    print(f"  SEED={SEED} | N_SPLITS={N_SPLITS} | cat_features={CAT_FEATURES}")
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

    assert list(X.columns) == list(X_test.columns), "train/test column mismatch"
    assert len(X_test) == EXPECTED_TEST_ROWS, (
        f"expected {EXPECTED_TEST_ROWS} test rows, got {len(X_test)}"
    )
    print(f"  features ({len(X.columns)}): {list(X.columns)}")
    print(f"  target rate train = {y.mean():.4f} (n_pos = {int(y.sum())})")

    print("\n[2/5] Train 5 folds")
    oof = np.zeros(len(X), dtype=np.float64)
    test_preds_per_fold = np.zeros((N_SPLITS, len(X_test)), dtype=np.float64)
    fold_rows: list[dict[str, float | int]] = []

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
        elapsed = time.time() - t0
        fold_rows.append(
            {
                "fold": fold_idx,
                "n_val": n_val,
                "frac_2023": frac_2023,
                "best_iter": best_iter,
                "auc": m["auc"],
                "brier": m["brier"],
                "logloss": m["logloss"],
                "seconds": round(elapsed, 1),
            }
        )
        print(
            f"  fold {fold_idx} | n_val={n_val:>6} | frac_2023={frac_2023:.3f} | "
            f"best_iter={best_iter:>4} | AUC={m['auc']:.4f} | "
            f"Brier={m['brier']:.4f} | logloss={m['logloss']:.4f} | "
            f"{elapsed:.1f}s"
        )

    print("\n[3/5] OOF aggregate")
    oof_m = compute_metrics(y, oof)
    auc_per_fold = np.array([fr["auc"] for fr in fold_rows])
    summary = {
        "oof_auc": oof_m["auc"],
        "oof_brier": oof_m["brier"],
        "oof_logloss": oof_m["logloss"],
        "per_fold_auc_mean": float(auc_per_fold.mean()),
        "per_fold_auc_std": float(auc_per_fold.std()),
        "per_fold_auc_min": float(auc_per_fold.min()),
        "per_fold_auc_max": float(auc_per_fold.max()),
    }
    for k, v in summary.items():
        print(f"  {k:>20} = {v:.4f}")

    print("\n[4/5] Test prediction = mean of 5 fold-models")
    test_pred = test_preds_per_fold.mean(axis=0)
    print(f"  test_pred shape    = {test_pred.shape}")
    print(f"  test_pred range    = [{test_pred.min():.4f}, {test_pred.max():.4f}]")
    print(f"  test_pred mean     = {test_pred.mean():.4f}")
    print(f"  test_pred median   = {float(np.median(test_pred)):.4f}")

    # Quality gates (f1-pitstops adapted from ALICE T7/T8)
    assert test_pred.shape == (EXPECTED_TEST_ROWS,), (
        f"shape gate: expected ({EXPECTED_TEST_ROWS},), got {test_pred.shape}"
    )
    assert not np.isnan(test_pred).any(), "NaN gate: NaN in test_pred"
    assert not np.isinf(test_pred).any(), "Inf gate: Inf in test_pred"
    assert (test_pred >= 0).all() and (test_pred <= 1).all(), "range gate: probs outside [0,1]"
    assert oof_m["auc"] > 0.5, f"signal gate: OOF AUC {oof_m['auc']:.4f} <= random (0.5)"
    print("  quality gates: OK (shape, NaN, Inf, range, AUC > 0.5)")

    print("\n[5/5] Write outputs to /kaggle/working/")
    sub_path = OUTPUT / "01_baseline.csv"
    pd.DataFrame({ID_COL: test_ids, TARGET: test_pred}).to_csv(sub_path, index=False)
    print(f"  wrote {sub_path} ({len(test_ids)} rows)")

    fold_csv = OUTPUT / "fold_metrics.csv"
    pd.DataFrame(fold_rows).to_csv(fold_csv, index=False)
    print(f"  wrote {fold_csv}")

    summary_path = OUTPUT / "oof_summary.json"
    summary["total_seconds"] = round(time.time() - t_start, 1)
    summary["git_ref"] = "f1d4a7e"
    summary["version"] = "v1"
    summary["seed"] = SEED
    summary["n_splits"] = N_SPLITS
    summary["cat_features"] = CAT_FEATURES
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"  wrote {summary_path}")

    print("\n" + "=" * 78)
    print(
        f"DONE - OOF AUC = {oof_m['auc']:.4f} | "
        f"per-fold AUC = {auc_per_fold.mean():.4f} +/- {auc_per_fold.std():.4f} | "
        f"total = {summary['total_seconds']}s"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
