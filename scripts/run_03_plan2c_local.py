"""Plan-2c (local, fast): drop the 5 high-card crosses that caused the CV mirage.

Ablation (postmortem) showed CatBoost CTR on high-card crosses inflates OOF +2.3pp
without LB transfer. This builds submit-02's FE MINUS those 5 crosses, disciplined
CatBoost (class weights, moderate iters), and writes a submission candidate locally
(no 2.4h GPU kernel). 5-fold OOF for honesty + full-train test prediction.

Run: .venv/Scripts/python.exe scripts/run_03_plan2c_local.py
Output: submissions/03_plan2c.csv  (+ prints OOF AUC, per-year)
"""

# ruff: noqa: C408  (dict() params kept for readability in this one-off script)
import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

SEED = 42
N_SPLITS = 5
TARGET = "PitNextLap"
ID = "id"
HIGH_CARD_DROP = [
    "Driver_Race",
    "Race_Compound_Stint",
    "Race_Year",
    "Driver_Compound",
    "Race_Compound",
]
AADI_PATHS = [
    "data/raw/f1_strategy_dataset_v4.csv",
    "data/external/f1_strategy_dataset_v4.csv",
]

spec = importlib.util.spec_from_file_location("k", "kernels/run_02_killer_fe/run_02_killer_fe.py")
k = importlib.util.module_from_spec(spec)
spec.loader.exec_module(k)


def main() -> None:
    train = pd.read_csv("data/raw/train.csv")
    test = pd.read_csv("data/raw/test.csv")
    train[TARGET] = train[TARGET].astype(int)
    aadi_path = next((p for p in AADI_PATHS if Path(p).exists()), None)
    original = None
    if aadi_path:
        original = pd.read_csv(aadi_path)
        original[TARGET] = original[TARGET].astype(int)
    print(
        f"train={len(train)} test={len(test)} aadi={'yes ' + aadi_path if aadi_path else 'NOT FOUND (skipping)'}"
    )

    train["IsOriginalData"] = "0"
    test["IsOriginalData"] = "0"
    train = k.add_cross_categoricals(k.add_domain_features(train))
    test = k.add_cross_categoricals(k.add_domain_features(test))
    if original is not None:
        original["IsOriginalData"] = "1"
        original = k.add_cross_categoricals(k.add_domain_features(original))

    comp_cols = [c for c in train.columns if c not in (ID, TARGET)]
    if original is not None:
        original = k.align_original(original, comp_cols)
        train, test, original, _ = k.add_frequency_encoding(train, test, original)
    else:
        # frequency over train+test only
        dummy = train.head(0).copy()
        train, test, _dummy, _ = k.add_frequency_encoding(train, test, dummy)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(skf.split(train, train[TARGET]))

    tu, su = k.compute_kfold_aggregates(train, test, splits, ["Year", "Race", "Driver"], "unit")
    ts, ss = k.compute_kfold_aggregates(
        train, test, splits, ["Year", "Race", "Driver", "Stint"], "stint"
    )
    train = pd.concat([train, tu, ts], axis=1)
    test = pd.concat([test, su, ss], axis=1)
    if original is not None:
        for c in list(tu.columns) + list(ts.columns):
            original[c] = np.nan

    # Plan-2c: DROP the 5 high-card crosses entirely (columns + from cat list).
    feature_cols = [c for c in train.columns if c not in (ID, TARGET) and c not in HIGH_CARD_DROP]
    feature_cols = [c for c in feature_cols if c in test.columns]
    cat_features = [c for c in k.ALL_CAT_FEATURES if c in feature_cols and c not in HIGH_CARD_DROP]
    for c in cat_features:
        train[c] = train[c].astype(str)
        test[c] = test[c].astype(str)
        if original is not None:
            original[c] = original[c].astype(str)
    print(f"features={len(feature_cols)} cats={len(cat_features)} (dropped {HIGH_CARD_DROP})")

    Xc = train[feature_cols]
    yc = train[TARGET].to_numpy()
    Xt = test[feature_cols]
    Xo = original[feature_cols] if original is not None else None
    yo = original[TARGET].to_numpy() if original is not None else None

    params = dict(
        iterations=2500,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=8.5,
        random_strength=0.65,
        bootstrap_type="Bayesian",
        bagging_temperature=0.45,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        task_type="CPU",
        early_stopping_rounds=200,
        random_seed=SEED,
        verbose=500,
        allow_writing_files=False,
    )

    oof = np.zeros(len(Xc))
    test_preds = np.zeros((N_SPLITS, len(Xt)))
    for i, (tri, vai) in enumerate(splits):
        Xtr = (
            pd.concat([Xc.iloc[tri], Xo], axis=0, ignore_index=True)
            if Xo is not None
            else Xc.iloc[tri]
        )
        ytr = np.concatenate([yc[tri], yo]) if yo is not None else yc[tri]
        m = CatBoostClassifier(**params)
        m.fit(
            Xtr,
            ytr,
            eval_set=(Xc.iloc[vai], yc[vai]),
            cat_features=cat_features,
            use_best_model=True,
        )
        oof[vai] = m.predict_proba(Xc.iloc[vai])[:, 1]
        test_preds[i] = m.predict_proba(Xt)[:, 1]
        print(
            f"  fold {i} AUC={roc_auc_score(yc[vai], oof[vai]):.4f} best_iter={m.get_best_iteration()}"
        )

    oof_auc = roc_auc_score(yc, oof)
    print(
        f"\nPlan-2c OOF AUC = {oof_auc:.4f}  (submit-02 OOF 0.9794 / LB 0.94296; submit-01 LB 0.94457)"
    )
    for y in (2022, 2023, 2024, 2025):
        mask = train["Year"].astype(int).to_numpy() == y
        if mask.sum() > 100 and 0 < yc[mask].mean() < 1:
            print(f"  year {y} AUC={roc_auc_score(yc[mask], oof[mask]):.4f}")

    test_pred = test_preds.mean(axis=0)
    sub = pd.DataFrame({ID: test[ID].to_numpy(), TARGET: test_pred})
    Path("submissions").mkdir(exist_ok=True)
    sub.to_csv("submissions/03_plan2c.csv", index=False)
    print(f"\nwrote submissions/03_plan2c.csv ({len(sub)} rows, mean={test_pred.mean():.4f})")


if __name__ == "__main__":
    main()
