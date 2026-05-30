"""submit-12 — RealMLP + (Y,R,D,Compound) single-column target-mean FE (D10 lever 1).

Why: blends are exhausted (D6 CatBoost / D8 TabPFN / D9 TabM all fail to beat RealMLP-v1
alone, LB 0.95407). The only path up the (dense) public LB is a STRONGER standalone
RealMLP. RealMLP-v1 only freq-encodes Driver/Race/Compound and target-encodes the two
combos (Race_Compound, Race_Year); it LACKS explicit out-of-fold target means by the key
single categoricals. Plan-2c's measured ablation showed single-column aggregates GENERALIZE
(the CatBoost mirage came from the 5 high-card CROSSES, NOT these). Adding them = genuinely
NEW information for the NN (pit-propensity per driver / race / year / compound), not just
variance reduction.

ONLY change vs submit-06 RealMLP: a second leak-free `TargetEncoder(cv=5)` over the RAW
single columns AGG_COLS, appended as numeric `_<col>_TGT` features. Same yekenot FE, same
StratifiedKFold(5, seed=42), same Aadi folding, same RealMLP_TD params. LB is the verdict
(OOF unreliable). seed=42.

Outputs (/kaggle/working/): oof_predictions.parquet (id,oof_pred,y,fold,Year),
    test_predictions.parquet (id,test_pred), realmlp_agg_submission.csv, realmlp_agg_summary.json.

Hardware: GPU T4 (torch cu128 OK on sm_75, NOT P100/sm_60 -> push with
    --accelerator NvidiaTeslaT4). enable_internet=true (pip pytabkit). version: submit-12.
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
# NEW vs submit-06: single-column out-of-fold target means (the generalizing aggregates;
# NOT the high-card crosses that caused the CatBoost mirage).
AGG_COLS = ["Driver", "Race", "Compound", "Year"]


def _find_csv(name: str) -> Path:
    matches = list(INPUT_ROOT.rglob(name))
    if not matches:
        raise FileNotFoundError(f"{name} not found under {INPUT_ROOT}")
    print(f"  found {name} at {matches[0]}")
    return matches[0]


category_map: dict = {}
num_cols: list[str] = []
cat_cols: list[str] = []
important_combos = [("Race", "Compound"), ("Race", "Year")]


def feature_engineering(df: pd.DataFrame, fit: bool = False):
    """Faithful reproduction of the yekenot RealMLP FE (identical to submit-06)."""
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

    for col in cat_cols + ["Year_cat_", "PitStop_cat_"]:
        count_name = f"_{col}_count" if col in cat_cols else f"_{col[:-1]}_count"
        if fit:
            count_map = df[col].value_counts()
            category_map[count_name] = count_map
        else:
            count_map = category_map[count_name]
        df[count_name] = df[col].map(count_map).fillna(0).astype("int32")

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


def _aggframe(df: pd.DataFrame) -> pd.DataFrame:
    """RAW single columns as strings, for leak-free target-mean encoding."""
    return pd.DataFrame({c: df[c].astype(str) for c in AGG_COLS}, index=df.index)


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
    import random

    import torch
    from pytabkit import RealMLP_TD_Classifier

    print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()}")
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
    print(f"  init cat={len(cat_cols)} num={len(num_cols)} | AGG_COLS={AGG_COLS}")

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
        "n_epochs": 16,
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
        "ls_eps": 0.0,
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

    agg_names = [f"_{c}_TGT" for c in AGG_COLS]

    for fold, ((tr_idx, val_idx), (or_tr_idx, _)) in enumerate(
        zip(skf.split(X, y), skf.split(orig, y_orig), strict=False), 0
    ):
        X_tr = pd.concat([X.iloc[tr_idx], orig.iloc[or_tr_idx]], axis=0).reset_index(drop=True)
        y_tr = pd.concat([y.iloc[tr_idx], y_orig.iloc[or_tr_idx]], axis=0).reset_index(drop=True)
        X_val, y_val = X.iloc[val_idx].copy(), y.iloc[val_idx]
        X_tst = X_test.copy()

        # (submit-06) leak-free target encoding on the 2 combos
        te = TargetEncoder(cv=N_SPLITS, smooth="auto", shuffle=True, random_state=SEED)
        te_names = [f"_{c}TE" for c in combo_names]
        X_tr[te_names] = te.fit_transform(X_tr[combo_names], y_tr)
        X_val[te_names] = te.transform(X_val[combo_names])
        X_tst[te_names] = te.transform(X_tst[combo_names])

        # (submit-12 NEW) leak-free target MEANS on raw single columns (Y,R,D,Compound)
        agg_te = TargetEncoder(cv=N_SPLITS, smooth="auto", shuffle=True, random_state=SEED)
        X_tr[agg_names] = agg_te.fit_transform(_aggframe(X_tr), y_tr)
        X_val[agg_names] = agg_te.transform(_aggframe(X_val))
        X_tst[agg_names] = agg_te.transform(_aggframe(X_tst))

        model = RealMLP_TD_Classifier(**params)
        model.fit(X_tr, y_tr, X_val, y_val)

        oof[val_idx] = model.predict_proba(X_val)[:, 1]
        fold_arr[val_idx] = fold
        test_pred += model.predict_proba(X_tst)[:, 1] / N_SPLITS
        auc = float(roc_auc_score(y_val, oof[val_idx]))
        fold_aucs.append(auc)
        print(f"  fold {fold} | AUC={auc:.5f} | elapsed={time.time() - t0:.0f}s")

        # incremental write (timeout-safe)
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
    print(f"\n  RealMLP+AGG OOF AUC = {oof_auc:.5f} | per-fold {np.mean(fold_aucs):.5f}")
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
        OUTPUT / "realmlp_agg_submission.csv", index=False
    )
    (OUTPUT / "realmlp_agg_summary.json").write_text(
        json.dumps(
            {
                "oof_auc": oof_auc,
                "per_fold_mean": float(np.mean(fold_aucs)),
                "per_fold": fold_aucs,
                "per_year": py,
                "agg_cols": AGG_COLS,
                "seed": SEED,
                "n_splits": N_SPLITS,
                "n_features": int(X.shape[1] + len(combo_names) + len(AGG_COLS)),
                "total_seconds": round(time.time() - t0, 1),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n  DONE | RealMLP+AGG OOF {oof_auc:.5f} | {time.time() - t0:.0f}s")
    print("  Next: probe alone on LB. Beats RealMLP-v1 0.95407? (OOF unreliable, LB the verdict)")


if __name__ == "__main__":
    main()
