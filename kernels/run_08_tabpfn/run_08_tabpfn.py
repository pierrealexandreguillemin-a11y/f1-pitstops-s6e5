"""submit-08 — TabPFN-3 (Prior Labs) third diversity family for the blend.

Why: submit-07 showed architectural diversity transfers to LB (CatBoost+RealMLP
blend = LB 0.95269, +0.186pp). TabPFN is a tabular FOUNDATION MODEL (in-context
transformer) — a THIRD architecture, and it uses RAW features (no FE), so it is
maximally decorrelated from CatBoost (heavy FE) and RealMLP (its own FE). A 3-way
isotonic blend can add further LB.

Recipe (standalone open-source tabpfn = TabPFN-3 weights, runs on Kaggle GPU, no API
token), per discussion 699096 (competitor CV: classifier default 0.95121):
    - RAW features (categoricals factorized to int codes, shared train/test mapping).
    - StratifiedKFold(5, shuffle, seed=42) — SAME folds as Plan-2c/RealMLP -> OOF aligns.
    - TabPFNClassifier(ignore_pretraining_limits=True, device='cuda', fit_mode='fit_with_cache').
    - Context capped at 100k stratified rows/fold for T4 memory+time safety (full 351k
      is feasible per competitor but risks OOM/timeout; 100k keeps it competent + reliable).
    - Batched predict (20k) to bound attention memory.

Outputs (/kaggle/working/): oof_predictions.parquet (id,oof_pred,y,fold,Year),
    test_predictions.parquet (id,test_pred), tabpfn_summary.json. Per-fold incremental
    write so a timeout still yields completed folds.

Hardware: GPU T4 (--accelerator NvidiaTeslaT4), enable_internet=true (pip tabpfn +
    auto-download weights from HF Prior-Labs). seed=42. version: submit-08. 2026-05-30.
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

INPUT_ROOT = Path("/kaggle/input")
OUTPUT = Path("/kaggle/working")
SEED = 42
N_SPLITS = 5
CONTEXT_CAP = 100_000
PREDICT_BATCH = 20_000
TARGET = "PitNextLap"
ID = "id"


def _find_csv(name: str) -> Path:
    matches = list(INPUT_ROOT.rglob(name))
    if not matches:
        raise FileNotFoundError(f"{name} not found under {INPUT_ROOT}")
    print(f"  found {name} at {matches[0]}")
    return matches[0]


def _batched_proba(model, X: np.ndarray) -> np.ndarray:  # noqa: N803 (sklearn X convention)
    out = []
    for s in range(0, len(X), PREDICT_BATCH):
        out.append(model.predict_proba(X[s : s + PREDICT_BATCH])[:, 1])
    return np.concatenate(out)


def main() -> None:
    t0 = time.time()
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tabpfn"])
    import torch
    from tabpfn import TabPFNClassifier

    print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()}")
    rng = np.random.default_rng(SEED)

    train = pd.read_csv(_find_csv("train.csv"))
    test = pd.read_csv(_find_csv("test.csv"))

    train_id = train[ID].to_numpy()
    test_id = test[ID].to_numpy()
    train_year = train["Year"].astype(int).to_numpy()
    y = train[TARGET].astype(int).to_numpy()
    feature_cols = [c for c in train.columns if c not in (ID, TARGET)]

    # RAW features; factorize object columns to int codes (shared train+test mapping).
    Xtr = train[feature_cols].copy()
    Xte = test[feature_cols].copy()
    for c in feature_cols:
        if Xtr[c].dtype == object or str(Xtr[c].dtype) == "category":
            cats = (
                pd.concat([Xtr[c].astype(str), Xte[c].astype(str)])
                .astype("category")
                .cat.categories
            )
            Xtr[c] = pd.Categorical(Xtr[c].astype(str), categories=cats).codes
            Xte[c] = pd.Categorical(Xte[c].astype(str), categories=cats).codes
    X = Xtr.to_numpy(dtype=np.float32)
    X_test = Xte.to_numpy(dtype=np.float32)
    print(f"  features={len(feature_cols)} | train={X.shape} test={X_test.shape}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    fold_arr = np.full(len(X), -1, dtype=np.int8)
    fold_aucs = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        # Stratified context cap for T4 memory/time safety.
        if len(tr_idx) > CONTEXT_CAP:
            pos = tr_idx[y[tr_idx] == 1]
            neg = tr_idx[y[tr_idx] == 0]
            frac = CONTEXT_CAP / len(tr_idx)
            n_pos = int(round(len(pos) * frac))
            n_neg = CONTEXT_CAP - n_pos
            ctx = np.concatenate(
                [rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)]
            )
            rng.shuffle(ctx)
        else:
            ctx = tr_idx

        model = TabPFNClassifier(
            ignore_pretraining_limits=True,
            device="cuda",
            fit_mode="fit_with_cache",
            random_state=SEED,
        )
        model.fit(X[ctx], y[ctx])

        oof[va_idx] = _batched_proba(model, X[va_idx])
        fold_arr[va_idx] = fold
        test_pred += _batched_proba(model, X_test) / N_SPLITS
        auc = float(roc_auc_score(y[va_idx], oof[va_idx]))
        fold_aucs.append(auc)
        print(f"  fold {fold} | ctx={len(ctx)} | AUC={auc:.5f} | elapsed={time.time() - t0:.0f}s")

        # Incremental write (timeout-safe): completed folds' OOF + running test mean.
        pd.DataFrame(
            {ID: train_id, "oof_pred": oof, "y": y, "fold": fold_arr, "Year": train_year}
        ).to_parquet(OUTPUT / "oof_predictions.parquet", index=False)
        pd.DataFrame({ID: test_id, "test_pred": test_pred * N_SPLITS / (fold + 1)}).to_parquet(
            OUTPUT / "test_predictions.parquet", index=False
        )

    # Final test pred = mean over all folds (already divided by N_SPLITS per fold).
    pd.DataFrame({ID: test_id, "test_pred": test_pred}).to_parquet(
        OUTPUT / "test_predictions.parquet", index=False
    )
    oof_auc = float(roc_auc_score(y, oof))
    per_year = {}
    for yr in sorted(np.unique(train_year)):
        msk = train_year == yr
        if msk.sum() >= 100 and 0 < y[msk].mean() < 1:
            per_year[int(yr)] = {"n": int(msk.sum()), "auc": float(roc_auc_score(y[msk], oof[msk]))}

    (OUTPUT / "tabpfn_summary.json").write_text(
        json.dumps(
            {
                "oof_auc": oof_auc,
                "per_fold_mean": float(np.mean(fold_aucs)),
                "per_fold": fold_aucs,
                "per_year": per_year,
                "context_cap": CONTEXT_CAP,
                "seed": SEED,
                "total_seconds": round(time.time() - t0, 1),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n  DONE | TabPFN OOF AUC = {oof_auc:.5f} | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
