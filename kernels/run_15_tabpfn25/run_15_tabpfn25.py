"""submit-15 (A2) — TabPFN UN-STARVED: rich yekenot FE (vs 14 raw in submit-08).

NOTE: pinned to v2 (tabpfn==2.0.9, proven no-gate) NOT 2.5 — the latest weights are
license-gated and crash in headless batch (memo D8). The dominant un-starve lever is the
RICH FE (14 -> ~40 features), not the 100k context (diminishing for ICL); 6x10k bagging
compensates for the 10k design limit. Original 2.5 rationale kept below for the record.

Why (research 2026-05-31, docs/research/2026-05-31-sota-and-audacious-levers.md):
    submit-08 TabPFN was DOUBLE-STARVED -> only 0.93318 (decorrelated ρ.919 but too weak to
    help the blend). Two starvations:
      (1) context capped at 10k (TabPFN-v2 design limit, arxiv 2502.17361),
      (2) only 14 RAW features (RealMLP got the rich yekenot FE).
    TabPFN-2.5 (Prior Labs, Nov 2025; `pip install tabpfn` now defaults to it) handles
    ~100k context x ~2000 features -> fixes BOTH. This kernel feeds it (i) the SAME rich
    yekenot FE as RealMLP (numeric arithmetic + counts + KBins + combo TargetEncoder, all
    factorized to numeric for the ICL transformer) and (ii) up to 50k stratified context with
    light bagging. If it climbs from 0.933 toward ~0.95 while staying a foundation-model
    architecture, it is a strong decorrelated partner / better final slot B than blend-09.

    Honest risk: TabPFN-2.5 weights may be license-gated in headless batch (our memo recorded
    TabPFN-3 -> TabPFNLicenseError). On a load/license failure this kernel exits with a clear
    diagnostic (fast, cheap) so we can fall back to v2 (2.0.9) + rich FE if needed.

Recipe: SAME StratifiedKFold(5, seed=42) + Aadi-in-fold + combo TargetEncoder as submit-06/08
    so OOF aligns by id. Per-fold + per-bag checkpoint (timeout/kill-safe).

Hardware: GPU T4 (--accelerator NvidiaTeslaT4). enable_internet=true (pip tabpfn + HF weights).
    seed=42. version: submit-15. written: 2026-05-31.
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
CONTEXT_CAP = 10_000  # TabPFN-v2 design limit (arxiv 2502.17361); 6 bags -> 60k coverage
N_BAGS = 6
PREDICT_BATCH = 20_000
TARGET = "PitNextLap"
ID = "id"

category_map: dict = {}
num_cols: list[str] = []
cat_cols: list[str] = []
important_combos = [("Race", "Compound"), ("Race", "Year")]


def _find_csv(name: str) -> Path:
    matches = list(INPUT_ROOT.rglob(name))
    if not matches:
        raise FileNotFoundError(f"{name} not found under {INPUT_ROOT}")
    print(f"  found {name} at {matches[0]}")
    return matches[0]


def feature_engineering(df: pd.DataFrame, fit: bool = False):
    """yekenot RealMLP FE (IDENTICAL to submit-06) — rich features for TabPFN to un-starve it."""
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

    combo_names_local = combo_names
    return df, combo_names_local


def main() -> None:
    t0 = time.time()
    # Pin tabpfn==2.0.9 (v2 weights, Apache-2.0, NO license-token gate). The "un-starve" lever
    # tested here is the RICH FE (14 -> ~40 features); the unpinned latest pulls TabPFN-2.5/3
    # weights that are license-gated and crash in headless batch (TabPFNLicenseError, memo D8).
    # 2.5's 100k context is the only thing 2.0.9 can't do (10k design limit) -> compensated by
    # 6 bags (60k coverage). Rich-FE is the dominant starvation fix; context is diminishing.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tabpfn==2.0.9"])
    import torch
    from tabpfn import TabPFNClassifier

    print("  tabpfn 2.0.9 imported (v2 weights, no gate)")

    print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()}")
    rng = np.random.default_rng(SEED)

    global num_cols, cat_cols
    train = pd.read_csv(_find_csv("train.csv"))
    test = pd.read_csv(_find_csv("test.csv"))
    orig = pd.read_csv(_find_csv("f1_strategy_dataset_v4.csv"))
    orig = orig.drop(["Normalized_TyreLife"], axis=1)
    y_orig = orig[TARGET].astype(int)
    orig = orig.drop([TARGET], axis=1)

    train_id = train[ID].to_numpy()
    test_id = test[ID].to_numpy()
    train_year = train["Year"].astype(int).to_numpy()
    y = train[TARGET].astype(int).to_numpy()
    X = train.drop([ID, TARGET], axis=1)
    X_test = test.drop([ID], axis=1)

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
    X, combo_names = feature_engineering(X, fit=True)
    X_test, _ = feature_engineering(X_test, fit=False)
    orig, _ = feature_engineering(orig, fit=False)

    # TabPFN wants a numeric matrix: factorize every object/str column to int codes (shared
    # train+test+orig mapping), keep numerics as-is. The rich FE (~40 cols) replaces the 14 raw.
    feat_cols = list(X.columns)
    for c in feat_cols:
        if X[c].dtype == object or str(X[c].dtype) == "category":
            allv = pd.concat([X[c].astype(str), X_test[c].astype(str), orig[c].astype(str)])
            cats = allv.astype("category").cat.categories
            X[c] = pd.Categorical(X[c].astype(str), categories=cats).codes
            X_test[c] = pd.Categorical(X_test[c].astype(str), categories=cats).codes
            orig[c] = pd.Categorical(orig[c].astype(str), categories=cats).codes
    print(f"  rich features for TabPFN = {len(feat_cols)} (was 14 raw in submit-08)")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    test_sum = np.zeros(len(X_test))
    n_test = 0
    fold_arr = np.full(len(X), -1, dtype=np.int8)
    fold_aucs: list[float] = []

    def _checkpoint() -> None:
        pd.DataFrame(
            {ID: train_id, "oof_pred": oof, "y": y, "fold": fold_arr, "Year": train_year}
        ).to_parquet(OUTPUT / "oof_predictions.parquet", index=False)
        pd.DataFrame({ID: test_id, "test_pred": test_sum / max(n_test, 1)}).to_parquet(
            OUTPUT / "test_predictions.parquet", index=False
        )

    def _batched_proba(model, Xn: np.ndarray) -> np.ndarray:  # noqa: N803
        out = []
        for s in range(0, len(Xn), PREDICT_BATCH):
            out.append(model.predict_proba(Xn[s : s + PREDICT_BATCH])[:, 1])
        return np.concatenate(out)

    orig_X = orig[feat_cols].to_numpy(dtype=np.float32)
    orig_y = y_orig.to_numpy()
    Xte_np = X_test[feat_cols].to_numpy(dtype=np.float32)

    for fold, ((tr_idx, va_idx), (or_tr_idx, _)) in enumerate(
        zip(skf.split(X, y), skf.split(orig_X, orig_y), strict=False)
    ):
        # Fold-aware TargetEncoder on combos (leak-free), then build numeric matrices.
        te = TargetEncoder(cv=N_SPLITS, smooth="auto", shuffle=True, random_state=SEED)
        Xtr_full = pd.concat([X.iloc[tr_idx], orig.iloc[or_tr_idx]], axis=0).reset_index(drop=True)
        ytr_full = np.concatenate([y[tr_idx], orig_y[or_tr_idx]])
        te_tr = te.fit_transform(Xtr_full[combo_names], ytr_full)
        te_va = te.transform(X.iloc[va_idx][combo_names])
        te_te = te.transform(X_test[combo_names])
        Xtr_mat = np.hstack(
            [Xtr_full[feat_cols].to_numpy(dtype=np.float32), te_tr.astype(np.float32)]
        )
        Xva_mat = np.hstack(
            [X.iloc[va_idx][feat_cols].to_numpy(dtype=np.float32), te_va.astype(np.float32)]
        )
        Xte_mat = np.hstack([Xte_np, te_te.astype(np.float32)])
        ytr_full_arr = ytr_full

        va_sum = np.zeros(len(va_idx))
        fold_arr[va_idx] = fold
        ctx_pool = np.arange(len(Xtr_mat))
        for b in range(N_BAGS):
            ctx = (
                _stratified_ctx_pool(ctx_pool, ytr_full_arr, rng)
                if len(ctx_pool) > CONTEXT_CAP
                else ctx_pool
            )
            model = TabPFNClassifier(device="cuda", random_state=SEED + b, n_estimators=4)
            model.fit(Xtr_mat[ctx], ytr_full_arr[ctx])
            va_sum += _batched_proba(model, Xva_mat)
            test_sum += _batched_proba(model, Xte_mat)
            n_test += 1
            oof[va_idx] = va_sum / (b + 1)
            _checkpoint()
            print(f"    fold {fold} bag {b} | ctx={len(ctx)} | elapsed={time.time() - t0:.0f}s")

        oof[va_idx] = va_sum / N_BAGS
        auc = float(roc_auc_score(y[va_idx], oof[va_idx]))
        fold_aucs.append(auc)
        print(f"  fold {fold} | AUC={auc:.5f} | elapsed={time.time() - t0:.0f}s")

    _checkpoint()
    oof_auc = float(roc_auc_score(y, oof))
    per_year = {}
    for yr in sorted(np.unique(train_year)):
        msk = train_year == yr
        if msk.sum() >= 100 and 0 < y[msk].mean() < 1:
            per_year[int(yr)] = {"n": int(msk.sum()), "auc": float(roc_auc_score(y[msk], oof[msk]))}
    pd.DataFrame({ID: test_id, TARGET: test_sum / max(n_test, 1)}).to_csv(
        OUTPUT / "tabpfn25_submission.csv", index=False
    )
    (OUTPUT / "tabpfn25_summary.json").write_text(
        json.dumps(
            {
                "oof_auc": oof_auc,
                "per_fold_mean": float(np.mean(fold_aucs)),
                "per_fold": fold_aucs,
                "per_year": per_year,
                "context_cap": CONTEXT_CAP,
                "n_bags": N_BAGS,
                "n_features": int(len(feat_cols) + len(combo_names)),
                "seed": SEED,
                "total_seconds": round(time.time() - t0, 1),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n  DONE | TabPFN-2.5 rich-FE OOF AUC = {oof_auc:.5f} | {time.time() - t0:.0f}s")


def _stratified_ctx_pool(pool: np.ndarray, yv: np.ndarray, rng) -> np.ndarray:
    """Stratified subsample of <= CONTEXT_CAP from a local index pool (advances rng each call)."""
    pos = pool[yv[pool] == 1]
    neg = pool[yv[pool] == 0]
    frac = CONTEXT_CAP / len(pool)
    n_pos = int(round(len(pos) * frac))
    n_neg = CONTEXT_CAP - n_pos
    ctx = np.concatenate(
        [
            rng.choice(pos, min(n_pos, len(pos)), replace=False),
            rng.choice(neg, min(n_neg, len(neg)), replace=False),
        ]
    )
    rng.shuffle(ctx)
    return ctx


if __name__ == "__main__":
    main()
