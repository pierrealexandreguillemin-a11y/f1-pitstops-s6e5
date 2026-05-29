"""submit-06 — RealMLP (PyTabKit) architectural-diversity partner for Plan-2c.

Why (discussion 699182, MEASURED on our data in submit-04):
    A GBDT+GBDT blend gave +0.000pp — rho(CatBoost,LightGBM)=0.958. Blending two
    boosting implementations does NOT add diversity; the gain comes from a model of
    a DIFFERENT architecture. RealMLP (a feed-forward NN with meta-tuned defaults,
    Holzmuller et al. NeurIPS 2024) is the documented lever. This kernel trains it
    so we can isotonic-blend it with the Plan-2c CatBoost (offline, tiny memory).

Recipe: faithful reproduction of the public yekenot RealMLP kernel (proven on this
    competition), which uses:
      - its OWN feature engineering (arithmetic interactions, floor-categorized
        numericals, count encoding, KBins discretization, Race_Compound / Race_Year
        combos + sklearn TargetEncoder) — DELIBERATELY different from Plan-2c's FE to
        maximise decorrelation,
      - StratifiedKFold(5, shuffle, seed=42) — IDENTICAL split to Plan-2c, so the OOF
        aligns row-for-row (we also key by id),
      - Aadi external data folded into each fold's training set (its own 5-fold split),
      - RealMLP_TD_Classifier meta-tuned defaults (no per-dataset HPO).

Outputs (/kaggle/working/):
    oof_predictions.parquet   id, oof_pred, y, fold, Year   (for the isotonic blend)
    test_predictions.parquet  id, test_pred
    realmlp_submission.csv     id, PitNextLap                (RealMLP-alone, optional probe)
    realmlp_summary.json       oof_auc, per-fold, per-year

Hardware: GPU T4 (torch cu128 OK on sm_75, NOT P100/sm_60 -> push with
    --accelerator NvidiaTeslaT4). enable_internet=true (pip install pytabkit).

seed=42. git_ref: realmlp-diversity. version: submit-06. written: 2026-05-30.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder

INPUT_ROOT = Path("/kaggle/input")
OUTPUT = Path("/kaggle/working")
SEED = 42
N_SPLITS = 5
TARGET = "PitNextLap"
ID = "id"


def _find_csv(name: str) -> Path:
    matches = list(INPUT_ROOT.rglob(name))
    if not matches:
        raise FileNotFoundError(f"{name} not found under {INPUT_ROOT}")
    print(f"  found {name} at {matches[0]}")
    return matches[0]


# Module-global fit state (yekenot pattern: feature_engineering mutates category_map).
category_map: dict = {}
num_cols: list[str] = []
cat_cols: list[str] = []
important_combos = [("Race", "Compound"), ("Race", "Year")]


def feature_engineering(df: pd.DataFrame, fit: bool = False):
    """Faithful reproduction of the yekenot RealMLP FE."""
    df["_LapNumber_/_RaceProgress"] = (df["LapNumber"] / (df["RaceProgress"] + 1e-6)).astype(
        "float32"
    )
    df["_TyreLife_/_LapNumber"] = (df["TyreLife"] / df["LapNumber"].clip(lower=1)).astype("float32")
    df["_LapTime (s)_*_Cumulative_Degradation"] = (
        df["LapTime (s)"] * df["Cumulative_Degradation"]
    ).astype("float32")
    df["_LapTime (s)_*_Cumulative_Degradation_abs"] = (
        df["LapTime (s)"] * df["Cumulative_Degradation"].abs()
    ).astype("float32")
    df["_LapTime (s)_/_Cumulative_Degradation_abs"] = (
        df["LapTime (s)"] / (df["Cumulative_Degradation"].abs() + 1e-6)
    ).astype("float32")

    # Categorize numericals (floor -> factorize -> str)
    for col in num_cols + ["_LapNumber_/_RaceProgress", "_TyreLife_/_LapNumber"]:
        cat_name = f"{col}_cat_" if col in num_cols else f"{col[1:]}_cat_"
        if fit:
            codes, uniques = np.floor(df[col]).factorize()
            category_map[col] = uniques
        else:
            uniques = category_map[col]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = np.floor(df[col]).map(code_map).fillna(-1).astype("int32")
        df[cat_name] = pd.Series(codes, index=df.index).astype(str)

    # Count encoding
    for col in cat_cols + ["Year_cat_", "PitStop_cat_"]:
        count_name = f"_{col}_count" if col in cat_cols else f"_{col[:-1]}_count"
        if fit:
            count_map = df[col].value_counts()
            category_map[count_name] = count_map
        else:
            count_map = category_map[count_name]
        df[count_name] = df[col].map(count_map).fillna(0).astype("int32")

    # Discretize numericals
    bin_config = {"RaceProgress": [200], "LapTime (s)": [7]}
    for col, bins_list in bin_config.items():
        for n_bins in bins_list:
            bin_name = f"{col}_{n_bins}_quantile_bin_"
            if fit:
                kb = KBinsDiscretizer(
                    n_bins=n_bins, encode="ordinal", strategy="quantile", subsample=None
                )
                binned = kb.fit_transform(df[[col]]).ravel().astype("int32")
                category_map[bin_name] = kb
            else:
                kb = category_map[bin_name]
                binned = kb.transform(df[[col]]).ravel().astype("int32")
            df[bin_name] = pd.Series(binned, index=df.index).astype(str)

    # Interaction categories
    combo_names = []
    for cols in important_combos:
        combo_name = "_".join(cols) + "_"
        combo_names.append(combo_name)
        combo_series = df[cols[0]].astype(str)
        for col in cols[1:]:
            combo_series = combo_series + "_" + df[col].astype(str)
        if fit:
            codes, uniques = pd.factorize(combo_series, sort=False)
            category_map[combo_name] = uniques
        else:
            uniques = category_map[combo_name]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = combo_series.map(code_map).fillna(-1).astype("int32")
        df[combo_name] = pd.Series(codes, index=df.index).astype(str)

    new_cat_cols = [c for c in df.columns if c.endswith("_")]
    new_num_cols = [c for c in df.columns if c.startswith("_")]
    return df, new_cat_cols, new_num_cols, combo_names


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


def main() -> None:
    t0 = time.time()
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pytabkit"])
    import torch
    from pytabkit import RealMLP_TD_Classifier

    print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()}")

    import random

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    global num_cols, cat_cols

    train = pd.read_csv(_find_csv("train.csv"))
    test = pd.read_csv(_find_csv("test.csv"))
    orig = pd.read_csv(_find_csv("f1_strategy_dataset_v4.csv"))

    orig = orig.drop(["Normalized_TyreLife"], axis=1)
    y_orig = orig[TARGET].astype(int)
    orig = orig.drop([TARGET], axis=1)
    train_id = train[ID]
    test_id = test[ID]
    train_year = train["Year"].astype(int).to_numpy()
    y = train[TARGET].astype(int)
    X = train.drop([ID, TARGET], axis=1)
    X_test = test.drop([ID], axis=1)

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
    print(f"  init cat={len(cat_cols)} num={len(num_cols)}")

    X, new_cat, new_num, combo_names = feature_engineering(X, fit=True)
    X_test, _, _, _ = feature_engineering(X_test, fit=False)
    orig, _, _, _ = feature_engineering(orig, fit=False)
    cat_cols += new_cat
    num_cols += new_num
    print(f"  prep features = {X.shape[1]} (cat+num) | combos={combo_names}")

    params = {
        "random_state": 42,
        "verbosity": 1,
        "val_metric_name": "1-auc_ovr",
        "n_ens": 20,
        "n_epochs": 5,
        "batch_size": 256,
        "use_early_stopping": False,
        "early_stopping_additive_patience": 10,
        "early_stopping_multiplicative_patience": 1,
        "lr": 0.019,
        "wd": 0.01,
        "sq_mom": 0.99,
        "lr_sched": "lin_cos_log_15",
        "first_layer_lr_factor": 0.25,
        "embedding_size": 6,
        "max_one_hot_cat_size": 18,
        "hidden_sizes": [512, 256, 128],
        "act": "silu",
        "p_drop": 0.05,
        "p_drop_sched": "invsqrtp1e-3",
        "plr_hidden_1": 16,
        "plr_hidden_2": 8,
        "plr_act_name": "gelu",
        "plr_lr_factor": 0.1151,
        "plr_sigma": 2.33,
        "ls_eps": 0.01,
        "ls_eps_sched": "sqrt_cos",
        "add_front_scale": False,
        "bias_init_mode": "neg-uniform-dynamic-2",
        "tfms": [
            "one_hot",
            "median_center",
            "robust_scale",
            "smooth_clip",
            "embedding",
            "l2_normalize",
        ],
    }

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    fold_arr = np.full(len(X), -1, dtype=np.int8)
    fold_aucs = []

    for fold, ((tr_idx, val_idx), (or_tr_idx, _)) in enumerate(
        zip(skf.split(X, y), skf.split(orig, y_orig), strict=False), 0
    ):
        X_tr = pd.concat([X.iloc[tr_idx], orig.iloc[or_tr_idx]], axis=0).reset_index(drop=True)
        y_tr = pd.concat([y.iloc[tr_idx], y_orig.iloc[or_tr_idx]], axis=0).reset_index(drop=True)
        X_val, y_val = X.iloc[val_idx].copy(), y.iloc[val_idx]
        X_tst = X_test.copy()

        # Target encoding on combos (cv inside training data only -> leak-free)
        te = TargetEncoder(cv=N_SPLITS, smooth="auto", shuffle=True, random_state=SEED)
        tr_enc = te.fit_transform(X_tr[combo_names], y_tr)
        val_enc = te.transform(X_val[combo_names])
        tst_enc = te.transform(X_tst[combo_names])
        te_names = [f"_{c}TE" for c in combo_names]
        X_tr[te_names] = tr_enc
        X_val[te_names] = val_enc
        X_tst[te_names] = tst_enc

        model = RealMLP_TD_Classifier(**params)
        model.fit(X_tr, y_tr, X_val, y_val)

        val_preds = model.predict_proba(X_val)[:, 1]
        oof[val_idx] = val_preds
        fold_arr[val_idx] = fold
        test_pred += model.predict_proba(X_tst)[:, 1] / N_SPLITS
        auc = float(roc_auc_score(y_val, val_preds))
        fold_aucs.append(auc)
        print(f"  fold {fold} | AUC={auc:.5f} | elapsed={time.time() - t0:.0f}s")

    oof_auc = float(roc_auc_score(y, oof))
    py = per_year_auc(y.to_numpy(), oof, train_year)
    print(
        f"\n  RealMLP OOF AUC = {oof_auc:.5f} | per-fold {np.mean(fold_aucs):.5f} +/- {np.std(fold_aucs):.5f}"
    )

    assert not np.isnan(test_pred).any() and (test_pred >= 0).all() and (test_pred <= 1).all()

    pd.DataFrame(
        {
            ID: train_id.to_numpy(),
            "oof_pred": oof,
            "y": y.to_numpy(),
            "fold": fold_arr,
            "Year": train_year,
        }
    ).to_parquet(OUTPUT / "oof_predictions.parquet", index=False)
    pd.DataFrame({ID: test_id.to_numpy(), "test_pred": test_pred}).to_parquet(
        OUTPUT / "test_predictions.parquet", index=False
    )
    pd.DataFrame({ID: test_id.to_numpy(), TARGET: test_pred}).to_csv(
        OUTPUT / "realmlp_submission.csv", index=False
    )
    (OUTPUT / "realmlp_summary.json").write_text(
        json.dumps(
            {
                "oof_auc": oof_auc,
                "per_fold_mean": float(np.mean(fold_aucs)),
                "per_fold": fold_aucs,
                "per_year": py,
                "seed": SEED,
                "n_splits": N_SPLITS,
                "n_features": int(X.shape[1] + len(combo_names)),
                "total_seconds": round(time.time() - t0, 1),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n  DONE | RealMLP OOF {oof_auc:.5f} | {time.time() - t0:.0f}s")
    print(
        "  Next: isotonic-blend with Plan-2c CatBoost OOF (offline). Probe if blend OOF > 0.95676."
    )


if __name__ == "__main__":
    main()
