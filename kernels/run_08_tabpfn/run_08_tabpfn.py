"""submit-08 — TabPFN-3 (Prior Labs) third diversity family for the blend.

Why: submit-07 showed architectural diversity transfers to LB (CatBoost+RealMLP
blend = LB 0.95269, +0.186pp). TabPFN is a tabular FOUNDATION MODEL (in-context
transformer) — a THIRD architecture, and it uses RAW features (no FE), so it is
maximally decorrelated from CatBoost (heavy FE) and RealMLP (its own FE). A 3-way
isotonic blend can add further LB.

Recipe (standalone open-source tabpfn = TabPFN-v2 weights, runs on Kaggle GPU, no API
token), per discussion 699096 (competitor CV: classifier default 0.95121) AND the
documented TabPFN-v2 large-data method (arxiv 2502.17361): the model's DESIGN LIMIT is
10k support rows; for >10k the canonical approach is "subsample to 10k support +
bagging ensemble" (like a random forest), NOT feeding a huge raw context.
    - RAW features (categoricals factorized to int codes, shared train/test mapping).
    - StratifiedKFold(5, shuffle, seed=42) — SAME folds as Plan-2c/RealMLP -> OOF aligns.
    - TabPFNClassifier(device='cuda', n_estimators=4) — WITHIN the 10k design limit, so
      no ignore_pretraining_limits (the prior 100k run exceeded the limit 10x -> the model
      warned "degraded performance" and ran ~10x slower; killed at 8650s. Fixed here.)
    - CONTEXT_CAP=10_000 stratified rows per bag; N_BAGS=4 independent stratified
      subsamples per fold, predictions averaged -> ~40k effective data, design-limit-safe.
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
CONTEXT_CAP = 10_000  # TabPFN-v2 design limit (arxiv 2502.17361). Do NOT exceed.
N_BAGS = 4  # independent stratified 10k subsamples per fold, averaged (bagging ensemble)
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
    # Pin TabPFN-v2 (Apache-2.0 weights, auto-download from HF Prior-Labs/TabPFN-v2-clf,
    # NO license-token gate). The current package defaults to TabPFN-3 whose weights are
    # license-gated and fail in non-interactive batch (TabPFNLicenseError). v2 avoids it.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tabpfn==2.0.9"])
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
    test_sum = np.zeros(len(X_test))  # running SUM of every completed bag's full-test proba
    n_test = 0  # number of completed (fold, bag) test predictions -> mean = test_sum / n_test
    fold_arr = np.full(len(X), -1, dtype=np.int8)
    fold_aucs = []

    def _checkpoint() -> None:
        """Write outputs after EVERY bag (timeout/kill-safe). All values validly scaled:
        completed folds' OOF is final; the in-progress fold's OOF is its bags-so-far mean;
        test = mean over all completed (fold,bag) predictions."""
        pd.DataFrame(
            {ID: train_id, "oof_pred": oof, "y": y, "fold": fold_arr, "Year": train_year}
        ).to_parquet(OUTPUT / "oof_predictions.parquet", index=False)
        pd.DataFrame({ID: test_id, "test_pred": test_sum / max(n_test, 1)}).to_parquet(
            OUTPUT / "test_predictions.parquet", index=False
        )

    def _stratified_ctx(tr_idx: np.ndarray) -> np.ndarray:
        """One stratified subsample of <= CONTEXT_CAP rows (rng advances each call -> bags differ)."""
        if len(tr_idx) <= CONTEXT_CAP:
            return tr_idx
        pos = tr_idx[y[tr_idx] == 1]
        neg = tr_idx[y[tr_idx] == 0]
        frac = CONTEXT_CAP / len(tr_idx)
        n_pos = int(round(len(pos) * frac))
        n_neg = CONTEXT_CAP - n_pos
        ctx = np.concatenate(
            [rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)]
        )
        rng.shuffle(ctx)
        return ctx

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        # Bagging ensemble: N_BAGS independent 10k subsamples, predictions averaged.
        va_sum = np.zeros(len(va_idx))  # raw sum over this fold's bags (divide by bags-done)
        fold_arr[va_idx] = fold  # mark rows so a mid-fold checkpoint is self-describing
        for b in range(N_BAGS):
            ctx = _stratified_ctx(tr_idx)
            model = TabPFNClassifier(device="cuda", random_state=SEED + b, n_estimators=4)
            model.fit(X[ctx], y[ctx])
            va_sum += _batched_proba(model, X[va_idx])
            test_sum += _batched_proba(model, X_test)
            n_test += 1
            oof[va_idx] = va_sum / (b + 1)  # valid partial average after each bag
            _checkpoint()  # kill-safe: write coherent outputs after EVERY bag
            print(f"    fold {fold} bag {b} | ctx={len(ctx)} | elapsed={time.time() - t0:.0f}s")

        oof[va_idx] = va_sum / N_BAGS  # finalized fold OOF
        auc = float(roc_auc_score(y[va_idx], oof[va_idx]))
        fold_aucs.append(auc)
        print(
            f"  fold {fold} | bags={N_BAGS} cap={CONTEXT_CAP} | AUC={auc:.5f} | elapsed={time.time() - t0:.0f}s"
        )

    # Final test pred = mean over ALL completed (fold, bag) predictions = test_sum / n_test.
    _checkpoint()
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
                "n_bags": N_BAGS,
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
