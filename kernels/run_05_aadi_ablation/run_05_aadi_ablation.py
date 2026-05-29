"""submit-05 diagnostic — Aadi external dataset ablation (WITH vs WITHOUT).

Question (from discussion 696380): 25.40% of the Aadi "original" dataset targets are
ALTERED vs the real FastF1 historical records (measured: 96134 matched, 24422 altered).
Our Plan-2c pipeline folds Aadi into every fold's training set => we may be injecting
~25% label noise. Does removing the external data help or hurt?

Controlled A/B in ONE kernel (same FE, same StratifiedKFold(42), same LightGBM):
    - Arm WITH-AADI  : exactly the Plan-2c pipeline (Aadi in frequency counts + Aadi rows
      concatenated into each fold's training set).
    - Arm NO-AADI    : Aadi removed entirely (frequency counts on comp train+test only,
      no Aadi rows in training). Measures the TOTAL contribution of the external data.
    The leak-free (Y,R,D)/(Y,R,D,Stint) aggregates are computed on competition train only
    (Aadi rows always get NaN) => identical across arms. Only the two Aadi touchpoints differ.

LightGBM (CPU on Kaggle, pip = CPU-only), DROP_HIGH_CARD_CROSSES=True (Plan-2c config),
1 seed for a fast diagnostic — we measure the OOF AUC DELTA, not a final model.

This is a DIAGNOSTIC, not a submission. Output:
    aadi_ablation.json   {with_aadi: {oof_auc, per_year}, no_aadi: {...}, delta}

Read order: this is the cheap proxy. If NO-AADI clearly beats WITH-AADI on LightGBM OOF,
the conclusive next step is a CatBoost-without-Aadi retrain + LB probe. If neutral/worse,
keep Aadi and pivot to RealMLP (architectural diversity) or consolidate.

seed=42. git_ref: aadi-ablation-diagnostic. version: submit-05-diagnostic. written: 2026-05-30.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Env-aware paths: same file runs as a Kaggle kernel AND locally (DRY).
# Kaggle mounts data under /kaggle/input and writes to /kaggle/working; locally we
# fall back to the repo's data/ dirs and write to submissions/.
_KAGGLE_IN = Path("/kaggle/input")
if _KAGGLE_IN.exists():
    INPUT_ROOTS = [_KAGGLE_IN]
    OUTPUT = Path("/kaggle/working")
else:
    _REPO = Path(__file__).resolve().parents[2]
    INPUT_ROOTS = [_REPO / "data" / "raw", _REPO / "data" / "external"]
    OUTPUT = _REPO / "submissions"
OUTPUT.mkdir(parents=True, exist_ok=True)
SEED = 42
N_SPLITS = 5
TARGET = "PitNextLap"
ID_COL = "id"
EPS = 1e-6

BASE_CAT = ["Driver", "Compound", "Race", "Year"]

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
# Plan-2c: drop the 5 high-card crosses (measured cause of the CatBoost CTR mirage).
_HIGH_CARD = {
    "Driver_Race",
    "Race_Compound_Stint",
    "Race_Year",
    "Driver_Compound",
    "Race_Compound",
}
CROSSES = [c for c in CROSSES if c[0] not in _HIGH_CARD]

BIN_FEATURES = ["RacePhase", "LapBin", "TyreLifeBin", "PositionBin"]
ALL_CAT_FEATURES = BASE_CAT + [c[0] for c in CROSSES] + BIN_FEATURES + ["IsOriginalData"]

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "n_estimators": 8000,
    "learning_rate": 0.02,
    "num_leaves": 63,
    "max_depth": 8,
    "min_child_samples": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "reg_lambda": 8.5,
    "reg_alpha": 0.1,
    "n_jobs": -1,
    "verbose": -1,
}
EARLY_STOPPING_ROUNDS = 200


# ---------- I/O ----------
def _find_csv(name: str) -> Path:
    for root in INPUT_ROOTS:
        matches = list(root.rglob(name))
        if matches:
            print(f"  found {name} at {matches[0]}")
            return matches[0]
    raise FileNotFoundError(f"{name} not found under {INPUT_ROOTS}")


def nan_audit(df: pd.DataFrame, split_name: str) -> None:
    for col in df.columns:
        if df[col].isna().mean() > 0.99:
            raise ValueError(f"{col} is {df[col].isna().mean():.0%} NaN on {split_name}")
    print(f"  [{split_name:>10}] {len(df):>7} rows x {len(df.columns)} cols")


# ---------- FE (faithful copy of run_02_killer_fe, Plan-2c config) ----------
def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rp_clip = df["RaceProgress"].clip(lower=EPS)
    df["EstimatedTotalLaps"] = (
        (df["LapNumber"] / rp_clip).replace([np.inf, -np.inf], np.nan).clip(1, 120)
    )
    df["LapsRemaining"] = (df["EstimatedTotalLaps"] - df["LapNumber"]).clip(lower=0)
    df["RemainingRaceProgress"] = 1.0 - df["RaceProgress"]
    df["LapProgress_x_LapNumber"] = df["LapNumber"] * df["RaceProgress"]
    lap_clip = df["LapNumber"].clip(lower=1)
    df["TyreAgeRatio"] = df["TyreLife"] / lap_clip
    df["TyreLifeMinusLap"] = df["TyreLife"] - df["LapNumber"]
    df["LapMinusTyreLife"] = df["LapNumber"] - df["TyreLife"]
    df["TyreAgeVsRace"] = df["TyreLife"] / df["EstimatedTotalLaps"].clip(lower=1)
    df["PitWindowPressure"] = df["TyreLife"] * df["RaceProgress"]
    df["StintPressure"] = df["Stint"] * df["TyreLife"]
    df["Is_First_Stint"] = (df["Stint"] == 1).astype(np.int8)
    df["Is_Late_Stint"] = (df["Stint"] >= 3).astype(np.int8)
    tl_clip = df["TyreLife"].clip(lower=1)
    df["DegPerTyreLap"] = df["Cumulative_Degradation"] / tl_clip
    df["Abs_Cumulative_Degradation"] = df["Cumulative_Degradation"].abs()
    df["DeltaAbs"] = df["LapTime_Delta"].abs()
    df["PositionPressure"] = df["Position"] * df["RaceProgress"]
    df["Abs_Position_Change"] = df["Position_Change"].abs()
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
    return df.replace([np.inf, -np.inf], np.nan)


def add_cross_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for name, cols in CROSSES:
        value = df[cols[0]].astype(str)
        for col in cols[1:]:
            value = value + "_" + df[col].astype(str)
        df[name] = value
    return df


def add_frequency_encoding(frames: list[pd.DataFrame]) -> tuple[list[pd.DataFrame], list[str]]:
    """Frequency-encode cat cols across the GIVEN frames combined (arm-dependent).

    WITH-AADI passes [train, test, original]; NO-AADI passes [train, test]. This is
    where one of the two Aadi touchpoints lives."""
    frames = [f.copy() for f in frames]
    freq_source_cols = [
        c
        for c in [*BASE_CAT, *[n for n, _ in CROSSES], *BIN_FEATURES]
        if all(c in f.columns for f in frames)
    ]
    total_rows = sum(len(f) for f in frames)
    new_cols: list[str] = []
    for col in freq_source_cols:
        combined = pd.concat([f[col].astype(str) for f in frames])
        counts = combined.value_counts()
        count_name, freq_name = f"{col}_count", f"{col}_freq"
        new_cols.extend([count_name, freq_name])
        for frame in frames:
            key = frame[col].astype(str)
            frame[count_name] = key.map(counts).fillna(0).astype(np.int32)
            frame[freq_name] = (frame[count_name] / total_rows).astype(np.float32)
    return frames, new_cols


def _group_aggregates(df: pd.DataFrame, group_cols: list[str], prefix: str) -> pd.DataFrame:
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


def compute_kfold_aggregates(comp_train, comp_test, splits, group_cols, prefix):
    """Leak-free k-fold aggregates on competition train only (Aadi-independent)."""
    n_train = len(comp_train)
    agg_col_names: list[str] | None = None
    buf: dict[str, np.ndarray] = {}
    for tr_idx, va_idx in splits:
        agg = _group_aggregates(comp_train.iloc[tr_idx], group_cols, prefix)
        val = comp_train.iloc[va_idx][group_cols].reset_index(drop=True)
        merged = val.merge(agg, on=group_cols, how="left")
        if agg_col_names is None:
            agg_col_names = [c for c in merged.columns if c not in group_cols]
            for c in agg_col_names:
                buf[c] = np.full(n_train, np.nan, dtype=np.float64)
        for c in agg_col_names:
            buf[c][va_idx] = merged[c].to_numpy()
    assert agg_col_names is not None
    train_agg = pd.DataFrame(buf, index=comp_train.index)
    agg_full = _group_aggregates(comp_train, group_cols, prefix)
    test_merged = (
        comp_test[group_cols].reset_index(drop=True).merge(agg_full, on=group_cols, how="left")
    )
    test_agg = pd.DataFrame(
        {c: test_merged[c].to_numpy() for c in agg_col_names}, index=comp_test.index
    )
    return train_agg, test_agg


def align_original(original: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    original = original.drop(columns=["Normalized_TyreLife"], errors="ignore").copy()
    for col in feature_cols:
        if col not in original.columns:
            original[col] = np.nan
    keep = feature_cols + ([TARGET] if TARGET in original.columns else [])
    return original[keep]


def per_year_auc(y_true, y_pred, year) -> dict:
    out = {}
    for y in sorted(np.unique(year)):
        mask = year == y
        if mask.sum() < 100 or y_true[mask].mean() in (0.0, 1.0):
            continue
        out[int(y)] = {
            "n": int(mask.sum()),
            "auc": float(roc_auc_score(y_true[mask], y_pred[mask])),
        }
    return out


# ---------- One ablation arm ----------
def run_arm(with_aadi: bool, train_raw, test_raw, original_raw, splits) -> dict:
    tag = "WITH-AADI" if with_aadi else "NO-AADI"
    print(
        f"\n{'=' * 70}\n[ARM {tag}] frequency+training {'include' if with_aadi else 'EXCLUDE'} Aadi\n{'=' * 70}"
    )
    train = add_cross_categoricals(add_domain_features(train_raw))
    test = add_cross_categoricals(add_domain_features(test_raw))

    if with_aadi:
        original = add_cross_categoricals(add_domain_features(original_raw))
        feat_cols0 = [c for c in train.columns if c not in (ID_COL, TARGET)]
        original = align_original(original, feat_cols0)
        (train, test, original), _ = add_frequency_encoding([train, test, original])
    else:
        original = None
        (train, test), _ = add_frequency_encoding([train, test])

    tr_unit, te_unit = compute_kfold_aggregates(
        train, test, splits, ["Year", "Race", "Driver"], "unit"
    )
    tr_stint, te_stint = compute_kfold_aggregates(
        train, test, splits, ["Year", "Race", "Driver", "Stint"], "stint"
    )
    train = pd.concat([train, tr_unit, tr_stint], axis=1)
    test = pd.concat([test, te_unit, te_stint], axis=1)

    feature_cols = [c for c in train.columns if c not in (ID_COL, TARGET)]
    cat_cols = [c for c in ALL_CAT_FEATURES if c in feature_cols]

    if with_aadi:
        for c in feature_cols:
            if c not in original.columns:
                original[c] = np.nan

    # shared category codes across all frames in this arm
    frames_for_codes = [train, test] + ([original] if with_aadi else [])
    for c in cat_cols:
        union = pd.concat([f[c].astype(str) for f in frames_for_codes]).unique()
        cd = pd.CategoricalDtype(categories=sorted(union))
        for f in frames_for_codes:
            f[c] = f[c].astype(str).astype(cd)

    X = train[feature_cols]
    y = train[TARGET].astype(int).to_numpy()
    if with_aadi:
        X_orig = original[feature_cols]
        y_orig = original[TARGET].astype(int).to_numpy()

    oof = np.zeros(len(X))
    fold_aucs = []
    for k, (tr_idx, va_idx) in enumerate(splits):
        if with_aadi:
            X_tr = pd.concat([X.iloc[tr_idx], X_orig], axis=0, ignore_index=True)
            y_tr = np.concatenate([y[tr_idx], y_orig])
        else:
            X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X.iloc[va_idx], y[va_idx]
        model = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=SEED)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="auc",
            categorical_feature=cat_cols,
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        oof[va_idx] = model.predict_proba(X_va)[:, 1]
        fold_aucs.append(float(roc_auc_score(y_va, oof[va_idx])))
        print(f"  [{tag}] fold {k} | AUC={fold_aucs[-1]:.5f} | best_iter~{model.best_iteration_}")

    oof_auc = float(roc_auc_score(y, oof))
    py = per_year_auc(y, oof, train["Year"].astype(int).to_numpy())
    n_tr_example = len(splits[0][0]) + (len(X_orig) if with_aadi else 0)
    print(
        f"  [{tag}] OOF AUC = {oof_auc:.5f} | per-fold {np.mean(fold_aucs):.5f} | n_train/fold~{n_tr_example}"
    )
    return {
        "oof_auc": oof_auc,
        "per_fold_mean": float(np.mean(fold_aucs)),
        "per_fold": fold_aucs,
        "per_year": py,
        "n_train_per_fold": int(n_tr_example),
    }


def main() -> None:
    t0 = time.time()
    np.random.seed(SEED)
    print("=" * 70)
    print("submit-05 diagnostic — Aadi external dataset ablation (WITH vs NO-AADI)")
    print("=" * 70)

    train_raw = pd.read_csv(_find_csv("train.csv"))
    test_raw = pd.read_csv(_find_csv("test.csv"))
    original_raw = pd.read_csv(_find_csv("f1_strategy_dataset_v4.csv"))
    nan_audit(train_raw, "train")
    nan_audit(test_raw, "test")
    nan_audit(original_raw, "original")
    train_raw[TARGET] = train_raw[TARGET].astype(int)
    original_raw[TARGET] = original_raw[TARGET].astype(int)
    train_raw["IsOriginalData"] = "0"
    test_raw["IsOriginalData"] = "0"
    original_raw["IsOriginalData"] = "1"

    # Aadi target-alteration rate (sanity echo of 696380, on overlap by feature key).
    print(
        f"  Aadi rows={len(original_raw)} | Aadi pit rate={original_raw[TARGET].mean():.4f} "
        f"vs comp train pit rate={train_raw[TARGET].mean():.4f}"
    )

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(skf.split(train_raw, train_raw[TARGET]))

    res_with = run_arm(True, train_raw, test_raw, original_raw, splits)
    # Incremental write — a timeout/crash before the 2nd arm still leaves arm-1 result.
    (OUTPUT / "aadi_ablation.json").write_text(
        json.dumps({"with_aadi": res_with, "no_aadi": None}, indent=2), encoding="utf-8"
    )
    res_no = run_arm(False, train_raw, test_raw, original_raw, splits)

    delta = res_no["oof_auc"] - res_with["oof_auc"]
    summary = {
        "with_aadi": res_with,
        "no_aadi": res_no,
        "delta_no_minus_with": delta,
        "verdict": (
            "NO-AADI better (Aadi hurts)"
            if delta > 0.0005
            else "WITH-AADI better (Aadi helps)"
            if delta < -0.0005
            else "neutral"
        ),
        "seed": SEED,
        "n_splits": N_SPLITS,
        "model": "LightGBM 1-seed (diagnostic proxy)",
        "total_seconds": round(time.time() - t0, 1),
    }
    (OUTPUT / "aadi_ablation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"WITH-AADI OOF = {res_with['oof_auc']:.5f}")
    print(f"NO-AADI   OOF = {res_no['oof_auc']:.5f}")
    print(f"DELTA (no - with) = {delta * 100:+.3f} pp  ->  {summary['verdict']}")
    print("=" * 70)
    print("Diagnostic only. If NO-AADI clearly wins -> CatBoost-without-Aadi retrain + LB probe.")


if __name__ == "__main__":
    main()
