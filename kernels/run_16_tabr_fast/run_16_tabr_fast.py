"""submit-16 (A1 fast) — RealTabR + early-stopping (faster at equal quality vs submit-13).

Same as submit-13 (100k subsample, 5 folds, numeric matrix) but with early stopping enabled
so we stop wasting epochs past the val peak (RealTabR_D already keeps best-epoch -> no quality
loss). Races submit-13 in the other GPU slot; take whichever finishes first with a clean OOF.

--- original submit-13 rationale ---
submit-13 (A1) — RealTabR (PyTabKit) the UNTESTED retrieval architecture.

Why (research 2026-05-31, docs/research/2026-05-31-sota-and-audacious-levers.md):
    Diversity coverage so far = GBDT (CatBoost ρ.958), foundation (TabPFN ρ.919 but weak),
    MLP (RealMLP engine + TabM ρ.989 CLONE). The one architectural family NEVER tried is
    RETRIEVAL / neighbour-based (TabR, ModernNCA). On TabArena it ranks at the top of deep
    methods, and its inductive bias (soft nearest-neighbours in a learned embedding space) is
    fundamentally different from a feed-forward MLP -> the best theoretical shot at a partner
    that is BOTH strong (~0.95) AND decorrelated (ρ well below TabM's 0.989). TabM failed only
    on correlation, not strength; a retrieval model keeps strength, breaks correlation.

    ModernNCA is not in pytabkit; TabR IS (RealTabR_D_Classifier) — same sklearn API as
    RealMLP, drop-in into the submit-06 scaffold (zero integration risk). RealTabR_D = the
    tuned-defaults variant (Holzmuller's improved TabR defaults).

Recipe: byte-for-byte the submit-06 pipeline (SAME yekenot FE, SAME StratifiedKFold(5,seed=42)
    so OOF aligns row-for-row by id, SAME Aadi external data, SAME combo TargetEncoder) — ONLY
    the model class changes. This isolates "what does a different architecture buy us" and keeps
    the OOF aligned for an isotonic blend with RealMLP-v1.

Outputs (/kaggle/working/): oof_predictions.parquet (id,oof_pred,y,fold,Year),
    test_predictions.parquet (id,test_pred), tabr_submission.csv (standalone probe),
    tabr_summary.json. Per-fold incremental write -> a timeout still yields completed folds
    (TabR runtime on ~430k rows is unverified — monitor fold 0).

Hardware: GPU T4 (--accelerator NvidiaTeslaT4 MANDATORY — P100/sm_60 breaks torch cu128).
    enable_internet=true (pip pytabkit). seed=42. version: submit-13. written: 2026-05-31.
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
    """Faithful reproduction of the yekenot RealMLP FE (IDENTICAL to submit-06)."""
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
    # TabR needs skorch (torch wrapper) — NOT pulled by pytabkit's base install; without it
    # RealTabR_D crashes at .fit() (ModuleNotFoundError 'skorch', confirmed v2 run).
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pytabkit", "skorch"])
    # TabR retrieves neighbours via faiss — NOT a pytabkit dependency (README: "conda only",
    # but pip wheels exist). Without it RealTabR_D crashes at .fit(). Install only if missing;
    # prefer the GPU wheel (fast retrieval over ~430k rows), fall back to CPU (reliable).
    try:
        import faiss  # noqa: F401

        print("  faiss already present")
    except ImportError:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "faiss-gpu-cu12"])
            print("  installed faiss-gpu-cu12")
        except subprocess.CalledProcessError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "faiss-cpu"])
            print("  installed faiss-cpu (GPU wheel unavailable)")
    import torch
    from pytabkit import RealTabR_D_Classifier

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
    print(f"  prep features = {X.shape[1]} (cat+num) | combos={combo_names}")

    # TabR one-hot-encodes EVERY categorical with NO cardinality cap -> 8GiB CUDA OOM on our
    # high-card combos / _cat_ cols (MEASURED, v3 crash). RealMLP capped one-hot at 18 + used
    # embeddings; TabR has no such guard. Fix: factorize ALL object cols to int codes (shared
    # X+test+orig mapping) and feed TabR a pure-NUMERIC float32 matrix -> no one-hot encoder is
    # built, only the retrieval path runs. Combos kept as codes for the fold-wise leak-free
    # TargetEncoder (sklearn TE treats int codes as categories -> identical grouping).
    obj_cols = X.select_dtypes(include=["object"]).columns.tolist()
    for c in obj_cols:
        cats = (
            pd.concat([X[c].astype(str), X_test[c].astype(str), orig[c].astype(str)])
            .astype("category")
            .cat.categories
        )
        X[c] = pd.Categorical(X[c].astype(str), categories=cats).codes.astype("int32")
        X_test[c] = pd.Categorical(X_test[c].astype(str), categories=cats).codes.astype("int32")
        orig[c] = pd.Categorical(orig[c].astype(str), categories=cats).codes.astype("int32")
    feat_cols = list(X.columns)
    print(f"  TabR numeric matrix = {len(feat_cols)} cols (all float, no one-hot)")

    # FASTER at EQUAL quality (submit-16): RealTabR_D keeps the best-epoch model (it gets X_val),
    # so epochs past the val peak only burn GPU. The submit-13 run trained 25+ epochs with no
    # stop. Enable early stopping (no hard n_epochs cap -> never cuts before the peak; stops only
    # once val plateaus) -> same best-epoch quality, fewer wasted epochs. These patience params
    # are pytabkit base-trainer kwargs (shared with RealMLP, which accepted them in submit-06).
    params = {
        "random_state": SEED,
        "n_threads": 4,
        "verbosity": 1,
        "use_early_stopping": True,
        "early_stopping_additive_patience": 8,
        "early_stopping_multiplicative_patience": 1,
    }

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    fold_arr = np.full(len(X), -1, dtype=np.int8)
    fold_aucs = []

    def _checkpoint() -> None:
        """Kill-safe: write coherent outputs after each fold (test = mean over folds done)."""
        done = max(len(fold_aucs), 1)
        pd.DataFrame(
            {
                ID: train_id.to_numpy(),
                "oof_pred": oof,
                "y": y.to_numpy(),
                "fold": fold_arr,
                "Year": train_year,
            }
        ).to_parquet(OUTPUT / "oof_predictions.parquet", index=False)
        pd.DataFrame({ID: test_id.to_numpy(), "test_pred": test_pred * N_SPLITS / done}).to_parquet(
            OUTPUT / "test_predictions.parquet", index=False
        )

    # TabR retrieval cost is O(N_candidates) per sample -> ~270s/epoch on the full ~430k train
    # (MEASURED v4: fold 0 not done after 70min, 5 folds would blow the deadline). Subsample the
    # per-fold training pool to MAX_TRAIN stratified rows -> ~63s/epoch, 5 folds finish with
    # margin. 100k candidates is plenty for retrieval; preserves the decorrelated inductive bias.
    MAX_TRAIN = 100_000
    sub_rng = np.random.default_rng(SEED)

    def _subsample(xm: np.ndarray, ym: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(ym) <= MAX_TRAIN:
            return xm, ym
        pos = np.where(ym == 1)[0]
        neg = np.where(ym == 0)[0]
        n_pos = int(round(len(pos) * MAX_TRAIN / len(ym)))
        idx = np.concatenate(
            [
                sub_rng.choice(pos, n_pos, replace=False),
                sub_rng.choice(neg, MAX_TRAIN - n_pos, replace=False),
            ]
        )
        sub_rng.shuffle(idx)
        return xm[idx], ym[idx]

    for fold, ((tr_idx, val_idx), (or_tr_idx, _)) in enumerate(
        zip(skf.split(X, y), skf.split(orig, y_orig), strict=False), 0
    ):
        X_tr = pd.concat([X.iloc[tr_idx], orig.iloc[or_tr_idx]], axis=0).reset_index(drop=True)
        y_tr = pd.concat([y.iloc[tr_idx], y_orig.iloc[or_tr_idx]], axis=0).reset_index(drop=True)
        X_val, y_val = X.iloc[val_idx].copy(), y.iloc[val_idx]

        te = TargetEncoder(cv=N_SPLITS, smooth="auto", shuffle=True, random_state=SEED)
        tr_enc = te.fit_transform(X_tr[combo_names], y_tr).astype("float32")
        val_enc = te.transform(X_val[combo_names]).astype("float32")
        tst_enc = te.transform(X_test[combo_names]).astype("float32")

        Xtr_mat = np.hstack([X_tr[feat_cols].to_numpy(np.float32), tr_enc])
        Xval_mat = np.hstack([X_val[feat_cols].to_numpy(np.float32), val_enc])
        Xtst_mat = np.hstack([X_test[feat_cols].to_numpy(np.float32), tst_enc])
        Xtr_mat, y_tr_arr = _subsample(Xtr_mat, y_tr.to_numpy())
        print(f"  fold {fold} | train subsampled to {len(y_tr_arr)} | t={time.time() - t0:.0f}s")

        model = RealTabR_D_Classifier(**params)
        model.fit(Xtr_mat, y_tr_arr, Xval_mat, y_val.to_numpy())

        val_preds = model.predict_proba(Xval_mat)[:, 1]
        oof[val_idx] = val_preds
        fold_arr[val_idx] = fold
        test_pred += model.predict_proba(Xtst_mat)[:, 1] / N_SPLITS
        auc = float(roc_auc_score(y_val, val_preds))
        fold_aucs.append(auc)
        print(f"  fold {fold} | AUC={auc:.5f} | elapsed={time.time() - t0:.0f}s")
        _checkpoint()

    oof_auc = float(roc_auc_score(y, oof))
    py = per_year_auc(y.to_numpy(), oof, train_year)
    print(
        f"\n  RealTabR OOF AUC = {oof_auc:.5f} | per-fold {np.mean(fold_aucs):.5f} +/- {np.std(fold_aucs):.5f}"
    )

    assert not np.isnan(test_pred).any() and (test_pred >= 0).all() and (test_pred <= 1).all()

    _checkpoint()
    pd.DataFrame({ID: test_id.to_numpy(), TARGET: test_pred}).to_csv(
        OUTPUT / "tabr_submission.csv", index=False
    )
    (OUTPUT / "tabr_summary.json").write_text(
        json.dumps(
            {
                "oof_auc": oof_auc,
                "per_fold_mean": float(np.mean(fold_aucs)),
                "per_fold": fold_aucs,
                "per_year": py,
                "seed": SEED,
                "n_splits": N_SPLITS,
                "total_seconds": round(time.time() - t0, 1),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n  DONE | RealTabR OOF {oof_auc:.5f} | {time.time() - t0:.0f}s")
    print("  Next: isotonic-blend with RealMLP-v1 OOF (offline). Probe alone vs 0.95407.")


if __name__ == "__main__":
    main()
