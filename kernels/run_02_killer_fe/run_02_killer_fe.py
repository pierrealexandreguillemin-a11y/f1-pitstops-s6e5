"""submit-02 "killer trick FE" kernel.

Self-contained Kaggle kernel implementing plan
docs/plans/2026-05-28-submit-02-plan.md.

Strategy 1 — beat submit-01 LB 0.94457 via incremental FE:
- (Y,R,D) and (Y,R,D,Stint) train-lookup aggregates (k-fold leak-free
  target encoding), per topic 699182 'defensive-pipeline-0.950-CV'.
  99.09% of test rows belong to a (Y,R,D) unit present in competition train.
- Aadi Gupta external dataset concatenated with IsOriginalData flag
  (real driver codes ALB/etc vs synthetic D109/etc - no Y,R,D overlap,
  Aadi rows get NaN aggregates, CatBoost handles natively).
- ~22 domain features + 9 cross-categoricals + frequency encoding.
- StratifiedKFold(5, shuffle, seed=42) replacing GroupKFold(Race+Year)
  which was pessimistic (CV-LB gap +1.52pp in submit-01).
- CatBoost depth=8 lr=0.018 l2_leaf_reg=8.5 early_stop=500 iter=11000.
- Per-fold + per-year AUC monitoring (defensive-pipeline finding).
- B0-B7 quality gates.

Repo traceability:
    git_ref:  bd5efb3
    version:  v1
    written:  2026-05-28
    target:   OOF 0.949-0.952, LB 0.950-0.952

Output (in /kaggle/working/):
    02_killer_fe.csv             submission (id, PitNextLap)
    fold_metrics.csv             per-fold + per-year metrics
    oof_predictions.parquet      id, oof_pred, y, fold (for submit-03 stacking)
    feature_importance.csv       feature, importance (CatBoost get_feature_importance)
    oof_summary.json             aggregate metrics + meta
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

# ---------- Configuration ----------
INPUT_ROOT = Path("/kaggle/input")
OUTPUT = Path("/kaggle/working")
SEED = 42
N_SPLITS = 5
EXPECTED_TEST_ROWS = 188_165
TARGET = "PitNextLap"
ID_COL = "id"
EPS = 1e-6

# Base categoricals (always-present, raw)
BASE_CAT = ["Driver", "Compound", "Race", "Year"]

# Numeric base columns
BASE_NUM = [
    "PitStop",
    "LapNumber",
    "Stint",
    "TyreLife",
    "Position",
    "LapTime (s)",
    "LapTime_Delta",
    "Cumulative_Degradation",
    "RaceProgress",
    "Position_Change",
]

# Crosses to build (name, [cols])
CROSSES = [
    ("Race_Year", ["Race", "Year"]),
    ("Compound_Stint", ["Compound", "Stint"]),
    ("Driver_Race", ["Driver", "Race"]),
    ("Driver_Compound", ["Driver", "Compound"]),
    ("Race_Compound", ["Race", "Compound"]),
    ("Race_Compound_Stint", ["Race", "Compound", "Stint"]),
    ("Compound_RacePhase", ["Compound", "RacePhase"]),
    ("Compound_TyreLifeBin", ["Compound", "TyreLifeBin"]),
    ("RacePhase_TyreLifeBin", ["RacePhase", "TyreLifeBin"]),
]

# Bin features (all categorical after pd.cut)
BIN_FEATURES = ["RacePhase", "LapBin", "TyreLifeBin", "PositionBin"]

# Cats that get frequency-encoded (BASE_CAT + crosses + bins)
ALL_CAT_FEATURES = BASE_CAT + [c[0] for c in CROSSES] + BIN_FEATURES + ["IsOriginalData"]

CATBOOST_PARAMS = {
    "iterations": 11000,
    "learning_rate": 0.018,
    "depth": 8,
    "l2_leaf_reg": 8.5,
    "random_strength": 0.65,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 0.45,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "task_type": "CPU",
    "early_stopping_rounds": 500,
    "verbose": 200,
    "allow_writing_files": False,
    "random_seed": SEED,
}


# ---------- I/O helpers ----------
def _walk_input(root: Path, max_depth: int = 4) -> None:
    """Diagnostic walk - print /kaggle/input/ tree at startup."""
    print(f"  walking {root} (max_depth={max_depth}):")
    if not root.exists():
        print(f"    !! {root} does not exist")
        return
    for path in sorted(root.rglob("*")):
        depth = len(path.relative_to(root).parts)
        if depth > max_depth:
            continue
        suffix = "/" if path.is_dir() else f"  ({path.stat().st_size} bytes)"
        print(f"    {'  ' * (depth - 1)}{path.name}{suffix}")


def _find_csv(name: str) -> Path:
    """Glob /kaggle/input/**/<name> - robust to any mount path."""
    matches = list(INPUT_ROOT.rglob(name))
    if not matches:
        raise FileNotFoundError(f"{name} not found under {INPUT_ROOT}")
    print(f"  found {name} at {matches[0]}")
    return matches[0]


def nan_audit(df: pd.DataFrame, split_name: str) -> None:
    """Raise if any column exceeds 99% NaN - ALICE pattern."""
    for col in df.columns:
        nan_pct = df[col].isna().mean()
        if nan_pct > 0.99:
            raise ValueError(f"{col} is {nan_pct:.0%} NaN on {split_name}")
    total_nan = int(df.isna().sum().sum())
    print(
        f"  [{split_name:>10}] {len(df):>7} rows x {len(df.columns)} cols | total NaN = {total_nan}"
    )


# ---------- FE: domain features ----------
def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ~17 numeric + 5 bin features in-place. Returns modified df."""
    df = df.copy()

    # Race progression
    rp_clip = df["RaceProgress"].clip(lower=EPS)
    df["EstimatedTotalLaps"] = (
        (df["LapNumber"] / rp_clip).replace([np.inf, -np.inf], np.nan).clip(1, 120)
    )
    df["LapsRemaining"] = (df["EstimatedTotalLaps"] - df["LapNumber"]).clip(lower=0)
    df["RemainingRaceProgress"] = 1.0 - df["RaceProgress"]
    df["LapProgress_x_LapNumber"] = df["LapNumber"] * df["RaceProgress"]

    # Tyre interactions
    lap_clip = df["LapNumber"].clip(lower=1)
    df["TyreAgeRatio"] = df["TyreLife"] / lap_clip
    df["TyreLifeMinusLap"] = df["TyreLife"] - df["LapNumber"]
    df["LapMinusTyreLife"] = df["LapNumber"] - df["TyreLife"]
    df["TyreAgeVsRace"] = df["TyreLife"] / df["EstimatedTotalLaps"].clip(lower=1)
    df["PitWindowPressure"] = df["TyreLife"] * df["RaceProgress"]

    # Stint
    df["StintPressure"] = df["Stint"] * df["TyreLife"]
    df["Is_First_Stint"] = (df["Stint"] == 1).astype(np.int8)
    df["Is_Late_Stint"] = (df["Stint"] >= 3).astype(np.int8)

    # Degradation / delta
    tl_clip = df["TyreLife"].clip(lower=1)
    df["DegPerTyreLap"] = df["Cumulative_Degradation"] / tl_clip
    df["Abs_Cumulative_Degradation"] = df["Cumulative_Degradation"].abs()
    df["DeltaAbs"] = df["LapTime_Delta"].abs()

    # Position
    df["PositionPressure"] = df["Position"] * df["RaceProgress"]
    df["Abs_Position_Change"] = df["Position_Change"].abs()

    # Bin features (cat strings)
    df["RacePhase"] = pd.cut(
        df["RaceProgress"],
        bins=[-np.inf, 0.20, 0.40, 0.60, 0.80, np.inf],
        labels=["P1", "P2", "P3", "P4", "P5"],
    ).astype(str)
    df["LapBin"] = pd.cut(
        df["LapNumber"],
        bins=[-np.inf, 5, 10, 20, 35, 50, np.inf],
        labels=["L_0_5", "L_6_10", "L_11_20", "L_21_35", "L_36_50", "L_51_plus"],
    ).astype(str)
    df["TyreLifeBin"] = pd.cut(
        df["TyreLife"],
        bins=[-np.inf, 3, 7, 12, 20, 30, np.inf],
        labels=["T_0_3", "T_4_7", "T_8_12", "T_13_20", "T_21_30", "T_31_plus"],
    ).astype(str)
    df["PositionBin"] = pd.cut(
        df["Position"],
        bins=[-np.inf, 3, 8, 14, np.inf],
        labels=["front", "upper_mid", "lower_mid", "back"],
    ).astype(str)

    # Replace inf with NaN (CatBoost prefers NaN over inf)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


# ---------- FE: cross-categoricals ----------
def add_cross_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for name, cols in CROSSES:
        value = df[cols[0]].astype(str)
        for col in cols[1:]:
            value = value + "_" + df[col].astype(str)
        df[name] = value
    return df


# ---------- FE: frequency encoding ----------
def add_frequency_encoding(
    train: pd.DataFrame, test: pd.DataFrame, original: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Compute count + freq for cat columns across train+test+original combined.

    Returns modified DataFrames + list of new freq feature column names.
    """
    train, test, original = train.copy(), test.copy(), original.copy()

    freq_source_cols = [
        c
        for c in [
            *BASE_CAT,
            *[name for name, _ in CROSSES],
            *BIN_FEATURES,
        ]
        if c in train.columns and c in test.columns and c in original.columns
    ]

    total_rows = len(train) + len(test) + len(original)
    new_cols: list[str] = []

    for col in freq_source_cols:
        combined = pd.concat(
            [
                train[col].astype(str),
                test[col].astype(str),
                original[col].astype(str),
            ]
        )
        counts = combined.value_counts()

        count_name = f"{col}_count"
        freq_name = f"{col}_freq"
        new_cols.extend([count_name, freq_name])

        for frame in (train, test, original):
            key = frame[col].astype(str)
            frame[count_name] = key.map(counts).fillna(0).astype(np.int32)
            frame[freq_name] = (frame[count_name] / total_rows).astype(np.float32)

    return train, test, original, new_cols


# ---------- FE: (Y,R,D) and (Y,R,D,Stint) k-fold leak-free aggregates ----------
def _group_aggregates(df: pd.DataFrame, group_cols: list[str], prefix: str) -> pd.DataFrame:
    """Compute aggregate stats for one groupby. Returns DataFrame indexed by group_cols."""
    grouped = df.groupby(group_cols)
    return pd.DataFrame(
        {
            f"{prefix}_max_TyreLife": grouped["TyreLife"].max(),
            f"{prefix}_n_rows": grouped.size(),
            f"{prefix}_max_Stint": grouped["Stint"].max(),
            f"{prefix}_LapNumber_max": grouped["LapNumber"].max(),
            f"{prefix}_LapNumber_min": grouped["LapNumber"].min(),
            f"{prefix}_LapTime_mean": grouped["LapTime (s)"].mean(),
            f"{prefix}_LapTime_std": grouped["LapTime (s)"].std(),
            f"{prefix}_PitStop_any": grouped["PitStop"].max().astype(np.float32),
            f"{prefix}_PitStop_sum": grouped["PitStop"].sum().astype(np.float32),
            f"{prefix}_PitNextLap_mean_OOF": grouped[TARGET].mean(),
        }
    ).reset_index()


def compute_kfold_aggregates(
    comp_train: pd.DataFrame,
    comp_test: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    group_cols: list[str],
    prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """K-fold leak-free aggregates for competition train + test.

    - For each fold (tr_idx, va_idx): compute aggregates on comp_train.iloc[tr_idx]
      and merge into comp_train.iloc[va_idx]. Repeat across 5 folds so every train
      row has aggregates computed from OTHER 4 folds.
    - For comp_test: compute aggregates from FULL comp_train (no leak since
      test has no target).

    Returns:
        train_agg: DataFrame indexed like comp_train, agg columns only (no group_cols)
        test_agg:  same for comp_test
    """
    print(f"  computing k-fold aggregates for groups={group_cols} prefix={prefix}")
    n_train = len(comp_train)
    agg_col_names: list[str] | None = None
    train_agg_buffer: dict[str, np.ndarray] = {}

    for _fold_idx, (tr_idx, va_idx) in enumerate(splits):
        train_fold = comp_train.iloc[tr_idx]
        val_fold = comp_train.iloc[va_idx][group_cols]

        agg = _group_aggregates(train_fold, group_cols, prefix)
        merged = val_fold.reset_index(drop=True).merge(agg, on=group_cols, how="left")

        if agg_col_names is None:
            agg_col_names = [c for c in merged.columns if c not in group_cols]
            for c in agg_col_names:
                train_agg_buffer[c] = np.full(n_train, np.nan, dtype=np.float64)

        for c in agg_col_names:
            train_agg_buffer[c][va_idx] = merged[c].to_numpy()

    assert agg_col_names is not None
    train_agg = pd.DataFrame(train_agg_buffer, index=comp_train.index)
    coverage_train = (~train_agg[f"{prefix}_n_rows"].isna()).mean() * 100
    print(f"    train coverage = {coverage_train:.2f}% (rows with at least one aggregate hit)")

    # Test: full train aggregates (target encoding safe - test has no labels in training)
    agg_full = _group_aggregates(comp_train, group_cols, prefix)
    test_merged = (
        comp_test[group_cols].reset_index(drop=True).merge(agg_full, on=group_cols, how="left")
    )
    test_agg = pd.DataFrame(
        {c: test_merged[c].to_numpy() for c in agg_col_names}, index=comp_test.index
    )
    coverage_test = (~test_agg[f"{prefix}_n_rows"].isna()).mean() * 100
    print(f"    test  coverage = {coverage_test:.2f}%")

    n_train_test_agg_cols = len(agg_col_names)
    print(f"    added {n_train_test_agg_cols} aggregate features (prefix={prefix})")
    return train_agg, test_agg


# ---------- Original (Aadi Gupta) alignment ----------
def align_original(original: pd.DataFrame, comp_train_feature_cols: list[str]) -> pd.DataFrame:
    """Drop Normalized_TyreLife, fill missing cols with NaN, reorder to match."""
    original = original.drop(columns=["Normalized_TyreLife"], errors="ignore").copy()
    for col in comp_train_feature_cols:
        if col not in original.columns:
            original[col] = np.nan
    # Keep only training-relevant columns + target
    keep = comp_train_feature_cols + ([TARGET] if TARGET in original.columns else [])
    return original[keep]


# ---------- Per-year breakdown ----------
def per_year_auc(y_true: np.ndarray, y_pred: np.ndarray, year: np.ndarray) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for y in sorted(np.unique(year)):
        mask = year == y
        if mask.sum() < 100:
            continue
        if y_true[mask].mean() in (0.0, 1.0):
            continue
        out[int(y)] = {
            "n": int(mask.sum()),
            "auc": float(roc_auc_score(y_true[mask], y_pred[mask])),
            "y_true_mean": float(y_true[mask].mean()),
            "y_pred_mean": float(y_pred[mask].mean()),
        }
    return out


# ---------- Main ----------
def main() -> None:
    np.random.seed(SEED)
    t_start = time.time()

    print("=" * 78)
    print("submit-02 'killer trick FE' kernel")
    print(f"  SEED={SEED} | N_SPLITS={N_SPLITS} | StratifiedKFold")
    print("=" * 78)

    # ---- 1. Load + sample_submission for B0 format check ----
    print("\n[1/7] Load + sample_submission B0 reference")
    _walk_input(INPUT_ROOT, max_depth=3)
    train_path = _find_csv("train.csv")
    test_path = _find_csv("test.csv")
    sample_path = _find_csv("sample_submission.csv")
    aadi_path = _find_csv("f1_strategy_dataset_v4.csv")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample_submission = pd.read_csv(sample_path)
    original = pd.read_csv(aadi_path)

    nan_audit(train, "train")
    nan_audit(test, "test")
    nan_audit(original, "original")

    # Cast target to int for sklearn AUC
    train[TARGET] = train[TARGET].astype(int)
    original[TARGET] = original[TARGET].astype(int)

    # Origin flag
    train["IsOriginalData"] = "0"
    test["IsOriginalData"] = "0"
    original["IsOriginalData"] = "1"

    # ---- 2. Domain features + crosses on all 3 splits ----
    print("\n[2/7] Add domain features + cross-categoricals")
    train = add_cross_categoricals(add_domain_features(train))
    test = add_cross_categoricals(add_domain_features(test))
    original = add_cross_categoricals(add_domain_features(original))
    print(f"  train shape after domain+crosses: {train.shape}")

    # ---- 3. Align original cols to comp_train + frequency encoding ----
    print("\n[3/7] Align original + frequency encoding")
    comp_train_feature_cols = [c for c in train.columns if c not in (ID_COL, TARGET)]
    original = align_original(original, comp_train_feature_cols)
    train, test, original, freq_cols = add_frequency_encoding(train, test, original)
    print(f"  added {len(freq_cols)} frequency features")

    # ---- 4. K-fold split on competition train ----
    print("\n[4/7] StratifiedKFold split (competition train only)")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(skf.split(train, train[TARGET]))
    for i, (_, va) in enumerate(splits):
        print(
            f"  fold {i}: n_val={len(va):>6}, val_target_rate={train.iloc[va][TARGET].mean():.4f}"
        )

    # ---- 5. (Y,R,D) + (Y,R,D,Stint) leak-free aggregates ----
    print("\n[5/7] Leak-free k-fold aggregates")
    train_unit_agg, test_unit_agg = compute_kfold_aggregates(
        train,
        test,
        splits,
        group_cols=["Year", "Race", "Driver"],
        prefix="unit",
    )
    train_stint_agg, test_stint_agg = compute_kfold_aggregates(
        train,
        test,
        splits,
        group_cols=["Year", "Race", "Driver", "Stint"],
        prefix="stint",
    )

    # Merge into train + test (aggregates only exist for competition train + test,
    # original gets NaN for these via reindex)
    train = pd.concat([train, train_unit_agg, train_stint_agg], axis=1)
    test = pd.concat([test, test_unit_agg, test_stint_agg], axis=1)
    for c in list(train_unit_agg.columns) + list(train_stint_agg.columns):
        original[c] = np.nan
    print(f"  train shape after aggregates: {train.shape}")
    print(f"  test  shape after aggregates: {test.shape}")
    print(f"  original shape after aggregates: {original.shape}")

    # ---- 6. Final feature alignment + CatBoost cat string casting ----
    print("\n[6/7] Cast cats to string for CatBoost, align feature set")
    feature_cols = [c for c in train.columns if c not in (ID_COL, TARGET)]
    assert set(feature_cols) == {c for c in test.columns if c != ID_COL}, (
        f"train/test feature mismatch: train={set(feature_cols) - set(test.columns)} test={set(test.columns) - set(feature_cols) - {ID_COL}}"
    )
    cat_features_present = [c for c in ALL_CAT_FEATURES if c in feature_cols]

    for c in cat_features_present:
        train[c] = train[c].astype(str)
        test[c] = test[c].astype(str)
        original[c] = original[c].astype(str)
    print(f"  total features = {len(feature_cols)}")
    print(f"  cat_features    = {len(cat_features_present)}: {cat_features_present}")

    X_train_comp = train[feature_cols]
    y_train_comp = train[TARGET].astype(int).values
    X_test = test[feature_cols]
    X_original = original[feature_cols]
    y_original = original[TARGET].astype(int).values
    test_ids = test[ID_COL]

    # ---- 7. CV training loop ----
    print("\n[7/7] Train 5 folds (comp_train_fold + original concat per fold)")
    oof = np.zeros(len(X_train_comp), dtype=np.float64)
    test_preds_per_fold = np.zeros((N_SPLITS, len(X_test)), dtype=np.float64)
    fold_rows: list[dict] = []
    per_year_records: list[dict] = []
    feature_importances: list[np.ndarray] = []

    for fold_idx, (tr_idx, va_idx) in enumerate(splits):
        t0 = time.time()
        X_tr = pd.concat([X_train_comp.iloc[tr_idx], X_original], axis=0, ignore_index=True)
        y_tr = np.concatenate([y_train_comp[tr_idx], y_original])
        X_va = X_train_comp.iloc[va_idx]
        y_va = y_train_comp[va_idx]

        model = CatBoostClassifier(**CATBOOST_PARAMS)
        model.fit(
            X_tr,
            y_tr,
            eval_set=(X_va, y_va),
            cat_features=cat_features_present,
            use_best_model=True,
        )

        va_pred = model.predict_proba(X_va)[:, 1]
        oof[va_idx] = va_pred
        test_preds_per_fold[fold_idx] = model.predict_proba(X_test)[:, 1]
        feature_importances.append(model.get_feature_importance())

        m = {
            "auc": float(roc_auc_score(y_va, va_pred)),
            "logloss": float(log_loss(y_va, np.clip(va_pred, 1e-7, 1 - 1e-7))),
            "brier": float(brier_score_loss(y_va, va_pred)),
        }
        years_va = train.iloc[va_idx]["Year"].astype(int).values
        per_year_fold = per_year_auc(y_va, va_pred, years_va)
        for year, stats in per_year_fold.items():
            per_year_records.append({"fold": fold_idx, "year": year, **stats})

        best_iter = int(model.get_best_iteration())
        elapsed = time.time() - t0
        fold_rows.append(
            {
                "fold": fold_idx,
                "n_train": int(len(X_tr)),
                "n_val": int(len(X_va)),
                "best_iter": best_iter,
                "auc": m["auc"],
                "brier": m["brier"],
                "logloss": m["logloss"],
                "seconds": round(elapsed, 1),
            }
        )
        print(
            f"  fold {fold_idx} | n_tr={len(X_tr):>6} n_va={len(X_va):>6} | "
            f"best_iter={best_iter:>4} | AUC={m['auc']:.4f} Brier={m['brier']:.4f} "
            f"logloss={m['logloss']:.4f} | {elapsed:.1f}s"
        )
        for year, stats in sorted(per_year_fold.items()):
            print(
                f"      year {year} | n={stats['n']:>6} | AUC={stats['auc']:.4f} | "
                f"y_true_mean={stats['y_true_mean']:.4f} | y_pred_mean={stats['y_pred_mean']:.4f}"
            )

    # ---- OOF aggregate ----
    print("\n[OOF aggregate]")
    oof_auc = float(roc_auc_score(y_train_comp, oof))
    oof_brier = float(brier_score_loss(y_train_comp, oof))
    oof_logloss = float(log_loss(y_train_comp, np.clip(oof, 1e-7, 1 - 1e-7)))
    auc_per_fold = np.array([r["auc"] for r in fold_rows])
    summary = {
        "oof_auc": oof_auc,
        "oof_brier": oof_brier,
        "oof_logloss": oof_logloss,
        "per_fold_auc_mean": float(auc_per_fold.mean()),
        "per_fold_auc_std": float(auc_per_fold.std()),
        "per_fold_auc_min": float(auc_per_fold.min()),
        "per_fold_auc_max": float(auc_per_fold.max()),
    }
    for k, v in summary.items():
        print(f"  {k:>22} = {v:.6f}")

    # OOF per-year
    print("\n[OOF per-year aggregate]")
    oof_per_year = per_year_auc(y_train_comp, oof, train["Year"].astype(int).values)
    for year, stats in sorted(oof_per_year.items()):
        print(
            f"  year {year} | n={stats['n']:>6} | AUC={stats['auc']:.4f} | "
            f"y_true_mean={stats['y_true_mean']:.4f} | y_pred_mean={stats['y_pred_mean']:.4f}"
        )

    # ---- Test prediction = mean of 5 fold-models ----
    print("\n[test prediction]")
    test_pred = test_preds_per_fold.mean(axis=0)
    print(f"  test_pred shape  = {test_pred.shape}")
    print(f"  test_pred range  = [{test_pred.min():.6f}, {test_pred.max():.6f}]")
    print(f"  test_pred mean   = {test_pred.mean():.6f}")
    print(f"  test_pred median = {float(np.median(test_pred)):.6f}")

    # ---- Quality gates ----
    print("\n[quality gates]")
    # B0: format match sample_submission
    assert list(sample_submission.columns) == [ID_COL, TARGET], (
        f"B0 fail: cols {list(sample_submission.columns)} != [{ID_COL}, {TARGET}]"
    )
    assert len(sample_submission) == EXPECTED_TEST_ROWS, (
        f"B0 fail: sample len {len(sample_submission)} != {EXPECTED_TEST_ROWS}"
    )
    assert (sample_submission[ID_COL].to_numpy() == test_ids.to_numpy()).all(), (
        "B0 fail: id sequence mismatch vs sample_submission"
    )
    print("  B0 format-match sample: OK")

    # B1-B5
    assert test_pred.shape == (EXPECTED_TEST_ROWS,), f"B1 fail: shape {test_pred.shape}"
    assert not np.isnan(test_pred).any(), "B2 fail: NaN in test_pred"
    assert not np.isinf(test_pred).any(), "B3 fail: Inf in test_pred"
    assert (test_pred >= 0).all() and (test_pred <= 1).all(), "B4 fail: range out of [0,1]"
    assert oof_auc > 0.5, f"B5 fail: OOF AUC {oof_auc:.4f} <= 0.5"
    print("  B1-B5 (shape, NaN, Inf, range, AUC>0.5): OK")

    # B6 per-year AUC > 0.85
    for year, stats in oof_per_year.items():
        assert stats["auc"] > 0.85, f"B6 fail: year {year} AUC {stats['auc']:.4f} <= 0.85"
    print("  B6 per-year AUC > 0.85: OK (all 4 years pass)")

    # B7 per-year calibration ±0.05 (groupby on int-cast Year - cat str cast above)
    test_years = test["Year"].astype(int).values
    train_year_rate = train.groupby(train["Year"].astype(int))[TARGET].mean()
    for year in sorted(np.unique(test_years)):
        mask = test_years == year
        if mask.sum() < 100:
            continue
        pred_mean = test_pred[mask].mean()
        train_rate = float(train_year_rate.loc[year])
        diff = abs(pred_mean - train_rate)
        ok = diff <= 0.05
        print(
            f"  B7 calibration year {year}: pred_mean={pred_mean:.4f} "
            f"train_rate={train_rate:.4f} diff={diff:.4f} {'OK' if ok else 'FAIL'}"
        )
        assert ok, (
            f"B7 fail: year {year} test_pred mean {pred_mean:.4f} vs train rate {train_rate:.4f}"
        )
    print("  B7 per-year calibration ±0.05: OK")

    # ---- Outputs ----
    print("\n[outputs]")
    sub = pd.DataFrame({ID_COL: test_ids, TARGET: test_pred})
    sub_path = OUTPUT / "02_killer_fe.csv"
    sub.to_csv(sub_path, index=False)
    print(f"  wrote {sub_path} ({len(sub)} rows)")

    fold_df = pd.DataFrame(fold_rows)
    (OUTPUT / "fold_metrics.csv").write_text(fold_df.to_csv(index=False), encoding="utf-8")
    print("  wrote fold_metrics.csv")

    per_year_df = pd.DataFrame(per_year_records)
    per_year_df.to_csv(OUTPUT / "per_year_metrics.csv", index=False)
    print("  wrote per_year_metrics.csv")

    # OOF parquet for submit-03 stacking
    fold_arr = _build_fold_array(splits, len(oof))
    oof_df = pd.DataFrame(
        {
            ID_COL: train[ID_COL].values,
            "oof_pred": oof,
            "y": y_train_comp,
            "fold": fold_arr,
            "Year": train["Year"].astype(int).values,
        }
    )
    oof_df.to_parquet(OUTPUT / "oof_predictions.parquet", index=False)
    print("  wrote oof_predictions.parquet")

    # Feature importance (mean across folds)
    fi_mean = np.mean(np.vstack(feature_importances), axis=0)
    fi_df = (
        pd.DataFrame({"feature": feature_cols, "importance": fi_mean})
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )
    fi_df.to_csv(OUTPUT / "feature_importance.csv", index=False)
    print("  wrote feature_importance.csv (top-10):")
    print(fi_df.head(10).to_string(index=False))

    # Summary JSON
    summary["per_year_oof"] = oof_per_year
    summary["total_seconds"] = round(time.time() - t_start, 1)
    summary["git_ref"] = "bd5efb3"
    summary["version"] = "v1"
    summary["seed"] = SEED
    summary["n_splits"] = N_SPLITS
    summary["n_features"] = len(feature_cols)
    summary["n_cat_features"] = len(cat_features_present)
    summary["expected_test_rows"] = EXPECTED_TEST_ROWS
    summary["b_gates_passed"] = ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7"]
    with (OUTPUT / "oof_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print("  wrote oof_summary.json")

    print("\n" + "=" * 78)
    print(
        f"DONE - OOF AUC = {oof_auc:.4f} | per-fold {auc_per_fold.mean():.4f} +/- "
        f"{auc_per_fold.std():.4f} | total = {summary['total_seconds']}s"
    )
    print("=" * 78)


def _build_fold_array(splits: list[tuple[np.ndarray, np.ndarray]], n: int) -> np.ndarray:
    """Build per-row fold assignment array for OOF parquet."""
    arr = np.full(n, -1, dtype=np.int8)
    for fold_idx, (_, va_idx) in enumerate(splits):
        arr[va_idx] = fold_idx
    assert (arr >= 0).all(), "fold assignment missing rows"
    return arr


if __name__ == "__main__":
    main()
