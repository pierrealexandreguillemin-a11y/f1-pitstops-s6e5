# ruff: noqa: E402
"""submit-04 — LightGBM diversity partner + blend with the Plan-2c CatBoost.

Goal (model-family diversity lever, documented in plan-ensemble):
    Plan-2c (CatBoost, LB 0.95083) is our best honest base. A second model from a
    DIFFERENT family (LightGBM = leaf-wise GBM with Fisher-style categorical splits,
    vs CatBoost's symmetric trees + ordered target statistics) decorrelates errors.
    A weighted blend of two decorrelated competent models typically lifts AUC by
    +0.1-0.3pp (documented winning lever).

Method — REUSE the exact Plan-2c feature pipeline (DRY, no FE drift):
    - Import the kernel module (kernels/run_02_killer_fe/run_02_killer_fe.py) and
      monkeypatch CROSSES / ALL_CAT_FEATURES to drop the 5 high-card crosses, i.e.
      the Plan-2c config (DROP_HIGH_CARD_CROSSES=True).
    - Replay the FE locally: domain features + crosses + frequency encoding +
      (Y,R,D) and (Y,R,D,Stint) leak-free k-fold aggregates.
    - StratifiedKFold(5, shuffle, seed=42) — IDENTICAL split to Plan-2c.
    - HARD ASSERT: reconstructed fold assignment == Plan-2c oof_predictions.parquet
      `fold` column. Catches any FE/order drift in one check (anti-leakage gate).
    - LightGBM 2-seed (42, 777) per fold, same train data (comp fold + Aadi original),
      same cat_features. Native categorical (pandas 'category' dtype, shared codes).

Blend:
    - Both OOF prediction vectors are genuine out-of-fold (no leakage).
    - Grid-search a single scalar weight w maximizing blend OOF AUC (1 scalar =
      negligible overfit; standard practice). Also try rank-average.
    - Apply w to test (Plan-2c test_pred + LightGBM test_pred), aligned by id.

Mirage guard (diagnostic-first):
    Plan-2c OOF 0.95676 sits ~0.6pp above its LB 0.95083 (trustworthy gap). If the
    LightGBM OOF lands far ABOVE ~0.957 (e.g. ~0.97), it is overfitting high-card
    cats differently => its OOF is a mini-mirage and would mislead blend weight
    selection. We MEASURE lgbm OOF and flag it. The real verdict is always the LB.

Inputs:
    data/raw/train.csv, data/raw/test.csv, data/raw/sample_submission.csv
    data/external/f1_strategy_dataset_v4.csv  (Aadi Gupta external)
    <plan2c_out>/oof_predictions.parquet, test_predictions.parquet  (CatBoost preds)

Output:
    submissions/04_lgbm_blend.csv      blended submission (id, PitNextLap)
    submissions/04_lgbm_oof.parquet    lgbm OOF (id, lgbm_oof, y, fold) for the record

Run:
    .venv/Scripts/python.exe scripts/run_04_lgbm_blend.py [--plan2c-dir DIR]

seed=42 everywhere. CPU (LightGBM pip = CPU-only).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parents[1]
KERNEL_PATH = REPO / "kernels" / "run_02_killer_fe" / "run_02_killer_fe.py"
DATA_RAW = REPO / "data" / "raw"
AADI_CSV = REPO / "data" / "external" / "f1_strategy_dataset_v4.csv"
SUBMISSIONS = REPO / "submissions"

SEED = 42
SEEDS = (42, 777)
N_SPLITS = 5
TARGET = "PitNextLap"
ID_COL = "id"
EXPECTED_TEST_ROWS = 188_165

HIGH_CARD_CROSSES = {
    "Driver_Race",
    "Race_Compound_Stint",
    "Race_Year",
    "Driver_Compound",
    "Race_Compound",
}

# LightGBM params — documented sensible defaults for ~540k rows binary AUC.
# Mirror CatBoost discipline where sensible (lr~0.018->0.02, l2~8.5->reg_lambda).
# num_leaves=63 << 2^max_depth-1 (255) for generalization; min_child_samples=100
# (LightGBM docs: "very important to prevent over-fitting in a leaf-wise tree").
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


def load_kernel_fe(drop_high_card: bool = True):
    """Import the kernel module and (optionally) drop the 5 high-card crosses.

    The kernel's FE functions read module-global CROSSES / ALL_CAT_FEATURES at call
    time, so reassigning them here propagates to add_cross_categoricals +
    add_frequency_encoding (Plan-2c config)."""
    spec = importlib.util.spec_from_file_location("killer_fe", KERNEL_PATH)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    sys.modules["killer_fe"] = m
    spec.loader.exec_module(m)
    if drop_high_card:
        m.CROSSES = [c for c in m.CROSSES if c[0] not in HIGH_CARD_CROSSES]
        m.ALL_CAT_FEATURES = (
            m.BASE_CAT + [c[0] for c in m.CROSSES] + m.BIN_FEATURES + ["IsOriginalData"]
        )
    print(f"  kernel FE loaded | CROSSES kept = {[c[0] for c in m.CROSSES]}")
    return m


def build_features(fe):
    """Replay the Plan-2c FE on train/test/original. Returns the assembled frames."""
    print("\n[FE] load raw")
    train = pd.read_csv(DATA_RAW / "train.csv")
    test = pd.read_csv(DATA_RAW / "test.csv")
    original = pd.read_csv(AADI_CSV)

    fe.nan_audit(train, "train")
    fe.nan_audit(test, "test")
    fe.nan_audit(original, "original")

    train[TARGET] = train[TARGET].astype(int)
    original[TARGET] = original[TARGET].astype(int)
    train["IsOriginalData"] = "0"
    test["IsOriginalData"] = "0"
    original["IsOriginalData"] = "1"

    print("[FE] domain features + cross-categoricals")
    train = fe.add_cross_categoricals(fe.add_domain_features(train))
    test = fe.add_cross_categoricals(fe.add_domain_features(test))
    original = fe.add_cross_categoricals(fe.add_domain_features(original))

    print("[FE] align original + frequency encoding")
    comp_train_feature_cols = [c for c in train.columns if c not in (ID_COL, TARGET)]
    original = fe.align_original(original, comp_train_feature_cols)
    train, test, original, freq_cols = fe.add_frequency_encoding(train, test, original)
    print(f"  +{len(freq_cols)} frequency features")

    return train, test, original


def reconstruct_splits(train: pd.DataFrame, plan2c_oof: pd.DataFrame):
    """StratifiedKFold(seed=42) split + HARD ASSERT it matches Plan-2c folds."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(skf.split(train, train[TARGET]))
    fold_arr = fe_fold_array(splits, len(train))

    # Plan-2c parquet preserves train id order; align its fold column by id.
    p = plan2c_oof.set_index(ID_COL).loc[train[ID_COL].to_numpy()]
    plan2c_fold = p["fold"].to_numpy()
    assert np.array_equal(fold_arr, plan2c_fold), (
        "FOLD MISMATCH: reconstructed StratifiedKFold(42) != Plan-2c fold column "
        "=> FE/order drift; blend OOF would be leaky/misaligned."
    )
    print(f"  split match OK: reconstructed folds == Plan-2c folds ({N_SPLITS} folds)")
    return splits


def fe_fold_array(splits, n: int) -> np.ndarray:
    arr = np.full(n, -1, dtype=np.int8)
    for k, (_, va) in enumerate(splits):
        arr[va] = k
    assert (arr >= 0).all()
    return arr


def add_aggregates(fe, train, test, splits):
    """(Y,R,D) + (Y,R,D,Stint) leak-free k-fold aggregates, merged into train/test."""
    print("\n[FE] leak-free k-fold aggregates")
    tr_unit, te_unit = fe.compute_kfold_aggregates(
        train, test, splits, group_cols=["Year", "Race", "Driver"], prefix="unit"
    )
    tr_stint, te_stint = fe.compute_kfold_aggregates(
        train, test, splits, group_cols=["Year", "Race", "Driver", "Stint"], prefix="stint"
    )
    train = pd.concat([train, tr_unit, tr_stint], axis=1)
    test = pd.concat([test, te_unit, te_stint], axis=1)
    return train, test


def prepare_matrices(fe, train, test, original):
    """Final feature alignment + categorical dtype with shared codes across splits."""
    feature_cols = [c for c in train.columns if c not in (ID_COL, TARGET)]
    assert set(feature_cols) == {c for c in test.columns if c != ID_COL}, "train/test mismatch"
    cat_cols = [c for c in fe.ALL_CAT_FEATURES if c in feature_cols]

    # original lacks aggregate cols -> NaN (matches the kernel)
    for c in feature_cols:
        if c not in original.columns:
            original[c] = np.nan

    # Shared category codes: build categories over the union so train/val/test/original
    # map identically (LightGBM splits on the integer codes).
    for c in cat_cols:
        union = pd.concat(
            [train[c].astype(str), test[c].astype(str), original[c].astype(str)]
        ).unique()
        cat_dtype = pd.CategoricalDtype(categories=sorted(union))
        train[c] = train[c].astype(str).astype(cat_dtype)
        test[c] = test[c].astype(str).astype(cat_dtype)
        original[c] = original[c].astype(str).astype(cat_dtype)

    print(f"  total features = {len(feature_cols)} | cat features = {len(cat_cols)}")
    return feature_cols, cat_cols


def train_lgbm(train, test, original, splits, feature_cols, cat_cols):
    """LightGBM 2-seed per fold. Returns (oof, test_pred, fold_aucs)."""
    X = train[feature_cols]
    y = train[TARGET].astype(int).to_numpy()
    X_test = test[feature_cols]
    X_orig = original[feature_cols]
    y_orig = original[TARGET].astype(int).to_numpy()

    oof = np.zeros(len(X), dtype=np.float64)
    test_pred = np.zeros(len(X_test), dtype=np.float64)
    fold_aucs = []

    print("\n[LGBM] train 5 folds x 2 seeds")
    for k, (tr_idx, va_idx) in enumerate(splits):
        X_tr = pd.concat([X.iloc[tr_idx], X_orig], axis=0, ignore_index=True)
        y_tr = np.concatenate([y[tr_idx], y_orig])
        X_va, y_va = X.iloc[va_idx], y[va_idx]

        va_seeds, test_seeds = [], []
        for seed in SEEDS:
            model = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=seed)
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="auc",
                categorical_feature=cat_cols,
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
            )
            va_seeds.append(model.predict_proba(X_va)[:, 1])
            test_seeds.append(model.predict_proba(X_test)[:, 1])
        va_pred = np.mean(va_seeds, axis=0)
        oof[va_idx] = va_pred
        test_pred += np.mean(test_seeds, axis=0) / N_SPLITS
        auc = roc_auc_score(y_va, va_pred)
        fold_aucs.append(auc)
        print(f"  fold {k} | AUC={auc:.5f} | best_iter~{model.best_iteration_}")
    return oof, test_pred, np.array(fold_aucs)


def search_blend_weight(y, cb_oof, lgb_oof):
    """Grid-search w for w*catboost + (1-w)*lgbm maximizing OOF AUC. Also rank-avg."""
    ws = np.linspace(0.0, 1.0, 101)
    aucs = [roc_auc_score(y, w * cb_oof + (1 - w) * lgb_oof) for w in ws]
    best_i = int(np.argmax(aucs))
    w_best, auc_best = float(ws[best_i]), float(aucs[best_i])

    cb_rank = pd.Series(cb_oof).rank().to_numpy()
    lgb_rank = pd.Series(lgb_oof).rank().to_numpy()
    rank_auc = roc_auc_score(y, 0.5 * cb_rank + 0.5 * lgb_rank)
    return w_best, auc_best, rank_auc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--plan2c-dir",
        default="C:/Users/pierr/AppData/Local/Temp/plan2c_out",
        help="dir containing Plan-2c oof_predictions.parquet + test_predictions.parquet",
    )
    args = ap.parse_args()
    plan2c_dir = Path(args.plan2c_dir)

    np.random.seed(SEED)
    print("=" * 78)
    print("submit-04 — LightGBM diversity partner + blend with Plan-2c CatBoost")
    print("=" * 78)

    cb_oof_df = pd.read_parquet(plan2c_dir / "oof_predictions.parquet")
    cb_test_df = pd.read_parquet(plan2c_dir / "test_predictions.parquet")
    cb_oof_auc = roc_auc_score(cb_oof_df["y"], cb_oof_df["oof_pred"])
    print(f"  Plan-2c CatBoost OOF AUC = {cb_oof_auc:.5f} (anchor; LB 0.95083)")

    fe = load_kernel_fe(drop_high_card=True)
    train, test, original = build_features(fe)
    splits = reconstruct_splits(train, cb_oof_df)
    train, test = add_aggregates(fe, train, test, splits)
    feature_cols, cat_cols = prepare_matrices(fe, train, test, original)

    lgb_oof, lgb_test, fold_aucs = train_lgbm(train, test, original, splits, feature_cols, cat_cols)

    y = train[TARGET].astype(int).to_numpy()
    lgb_oof_auc = roc_auc_score(y, lgb_oof)
    print("\n[LGBM OOF]")
    print(
        f"  OOF AUC = {lgb_oof_auc:.5f} | per-fold {fold_aucs.mean():.5f} +/- {fold_aucs.std():.5f}"
    )
    print(f"  Brier   = {brier_score_loss(y, lgb_oof):.5f}")

    # Mirage guard (diagnostic-first): flag if lgbm OOF is far above the trustworthy band.
    if lgb_oof_auc > 0.965:
        print(
            f"  !! WARN: LightGBM OOF {lgb_oof_auc:.4f} >> Plan-2c anchor 0.957 — likely a "
            "categorical mirage; blend weight on OOF may mislead. LB is the only verdict."
        )

    # Align CatBoost OOF/test to the same id order as our train/test.
    cb_oof = cb_oof_df.set_index(ID_COL).loc[train[ID_COL].to_numpy(), "oof_pred"].to_numpy()
    cb_test = cb_test_df.set_index(ID_COL).loc[test[ID_COL].to_numpy(), "test_pred"].to_numpy()
    assert np.array_equal(
        cb_oof_df.set_index(ID_COL).loc[train[ID_COL].to_numpy(), "y"].to_numpy(), y
    ), "y mismatch between CatBoost OOF and local train"

    w, blend_auc, rank_auc = search_blend_weight(y, cb_oof, lgb_oof)
    corr = float(np.corrcoef(cb_oof, lgb_oof)[0, 1])
    print("\n[BLEND]")
    print(f"  OOF corr(catboost, lgbm)     = {corr:.4f}  (lower = more diverse)")
    print(f"  CatBoost alone OOF AUC       = {cb_oof_auc:.5f}")
    print(f"  LightGBM alone OOF AUC       = {lgb_oof_auc:.5f}")
    print(f"  best linear blend w={w:.2f}   = {blend_auc:.5f}  (w*catboost + (1-w)*lgbm)")
    print(f"  rank-average (0.5/0.5)       = {rank_auc:.5f}")
    gain = blend_auc - cb_oof_auc
    print(f"  blend gain vs CatBoost alone = {gain * 100:+.3f} pp (OOF)")

    # Final submission: best linear blend on test, aligned by id.
    blend_test = w * cb_test + (1 - w) * lgb_test
    assert blend_test.shape == (EXPECTED_TEST_ROWS,)
    assert not np.isnan(blend_test).any() and not np.isinf(blend_test).any()
    assert (blend_test >= 0).all() and (blend_test <= 1).all()

    SUBMISSIONS.mkdir(exist_ok=True)
    sub = pd.DataFrame({ID_COL: test[ID_COL].to_numpy(), TARGET: blend_test})
    sub_path = SUBMISSIONS / "04_lgbm_blend.csv"
    sub.to_csv(sub_path, index=False)
    print(f"\n  wrote {sub_path} ({len(sub)} rows, mean={blend_test.mean():.4f})")

    lgb_oof_out = pd.DataFrame(
        {
            ID_COL: train[ID_COL].to_numpy(),
            "lgbm_oof": lgb_oof,
            "y": y,
            "fold": fe_fold_array(splits, len(y)),
        }
    )
    lgb_oof_out.to_parquet(SUBMISSIONS / "04_lgbm_oof.parquet", index=False)
    print(f"  wrote {SUBMISSIONS / '04_lgbm_oof.parquet'}")

    print("\n" + "=" * 78)
    print(
        f"DONE | CatBoost {cb_oof_auc:.5f} | LightGBM {lgb_oof_auc:.5f} | "
        f"blend(w={w:.2f}) {blend_auc:.5f} ({gain * 100:+.3f}pp OOF) | corr {corr:.3f}"
    )
    print("  Ship gate: submit ONLY if you expect LB > 0.95083 AND user OK. LB is the verdict.")
    print("=" * 78)


if __name__ == "__main__":
    main()
