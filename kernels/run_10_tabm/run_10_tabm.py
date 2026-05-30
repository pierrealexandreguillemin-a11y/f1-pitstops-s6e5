"""submit-10 — TabM (PyTabKit) second strong-NN diversity partner for RealMLP.

Why (measured this session):
    - RealMLP-v1 ALONE = LB 0.95407 (our best). RealMLP-v2 (config change) = 0.95398,
      corr 0.999 with v1 -> tuning RealMLP does NOT decorrelate it.
    - TabPFN (submit-08) was decorrelated (corr 0.919) but WEAK (LB 0.93318) -> its
      blend (submit-09) could not beat RealMLP-alone (0.95399 < 0.95407): a weak model
      gets ~0 weight.
    - The real lever = a SECOND strong NN of a DIFFERENT architecture. TabM (Gorishniy
      et al., ICLR 2025, arxiv 2410.24210) is a parameter-efficient ensemble of MLPs,
      frequently the best tabular DL model (>= RealMLP on TabArena). Being ~as strong as
      RealMLP AND architecturally different, a RealMLP+TabM blend is the classic diversity
      gain that TabPFN could not provide. TabM-alone may also be a new best single.

Recipe: IDENTICAL pipeline to submit-06 RealMLP (same yekenot FE, same
    StratifiedKFold(5, shuffle, seed=42) so the OOF aligns row-for-row for the blend,
    same Aadi external data folded in, same per-fold leak-free TargetEncoder on combos).
    ONLY the model is swapped: RealMLP_TD_Classifier -> TabM_D_Classifier with library
    tuned defaults (the "_D" variant is already meta-tuned; we do NOT inject RealMLP's
    own hyperparameters, which are architecture-specific). val_metric_name='1-auc_ovr'
    matches the competition metric (AUC). device auto-detects cuda (same as RealMLP).

Outputs (/kaggle/working/):
    oof_predictions.parquet   id, oof_pred, y, fold, Year   (for the isotonic blend)
    test_predictions.parquet  id, test_pred
    tabm_submission.csv        id, PitNextLap                (TabM-alone probe)
    tabm_summary.json          oof_auc, per-fold, per-year

Hardware: GPU T4 (torch cu128 OK on sm_75, NOT P100/sm_60 -> push with
    --accelerator NvidiaTeslaT4). enable_internet=true (pip install pytabkit).

seed=42. version: submit-10. written: 2026-05-30.
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
    """Faithful reproduction of the yekenot RealMLP FE (shared with submit-06 -> same
    feature space, so RealMLP and TabM differ ONLY by architecture)."""
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
    from pytabkit import TabM_D_Classifier

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

    # TabM_D = library tuned defaults (already meta-tuned). We pass ONLY safe shared
    # params: AUC validation metric (our target), seed, verbosity. device auto-detects
    # cuda exactly like RealMLP_TD did. No RealMLP-specific hyperparameters (invalid here).
    params = {
        "random_state": SEED,
        "val_metric_name": "1-auc_ovr",
        "verbosity": 1,
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

        model = TabM_D_Classifier(**params)
        model.fit(X_tr, y_tr, X_val, y_val)

        val_preds = model.predict_proba(X_val)[:, 1]
        oof[val_idx] = val_preds
        fold_arr[val_idx] = fold
        test_pred += model.predict_proba(X_tst)[:, 1] / N_SPLITS
        auc = float(roc_auc_score(y_val, val_preds))
        fold_aucs.append(auc)
        print(f"  fold {fold} | AUC={auc:.5f} | elapsed={time.time() - t0:.0f}s")

        # Incremental write (timeout/restart-safe): completed folds' OOF + running test mean.
        pd.DataFrame(
            {
                ID: train_id.to_numpy(),
                "oof_pred": oof,
                "y": y.to_numpy(),
                "fold": fold_arr,
                "Year": train_year,
            }
        ).to_parquet(OUTPUT / "oof_predictions.parquet", index=False)
        pd.DataFrame(
            {ID: test_id.to_numpy(), "test_pred": test_pred * N_SPLITS / (fold + 1)}
        ).to_parquet(OUTPUT / "test_predictions.parquet", index=False)

    oof_auc = float(roc_auc_score(y, oof))
    py = per_year_auc(y.to_numpy(), oof, train_year)
    print(
        f"\n  TabM OOF AUC = {oof_auc:.5f} | per-fold {np.mean(fold_aucs):.5f} +/- {np.std(fold_aucs):.5f}"
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
        OUTPUT / "tabm_submission.csv", index=False
    )
    (OUTPUT / "tabm_summary.json").write_text(
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
    print(f"\n  DONE | TabM OOF {oof_auc:.5f} | {time.time() - t0:.0f}s")
    print("  Next: isotonic-blend with RealMLP-v1 OOF (offline). Probe if blend LB > 0.95407.")


if __name__ == "__main__":
    main()
