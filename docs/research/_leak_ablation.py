"""Localize the submit-02 OOF->LB gap: is it CatBoost's categorical CTR?

Measured so far (LightGBM):
  base                       OOF 0.9209
  base + all submit-02 aggs  OOF 0.9318  (generalizes, ~ LB 0.943)
submit-02 CatBoost OOF 0.9794, LB 0.94296.

Hypothesis: the gap is CatBoost CTR on high-cardinality categorical crosses
overfitting the random-row CV. Test: CatBoost 5-fold OOF, cats vs numeric-only.
"""

# ruff: noqa: E402, C408  (scratch research script: filterwarnings before imports, dict() params)
import warnings

warnings.filterwarnings("ignore")
import importlib.util

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

spec = importlib.util.spec_from_file_location("k", "kernels/run_02_killer_fe/run_02_killer_fe.py")
k = importlib.util.module_from_spec(spec)
spec.loader.exec_module(k)

tr = pd.read_csv("data/raw/train.csv")
tr["PitNextLap"] = tr["PitNextLap"].astype(int)
y = tr["PitNextLap"].to_numpy()
splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(tr, y))

# Build the exact submit-02 feature frame (domain + crosses + aggregates), comp-train only.
trf = k.add_cross_categoricals(k.add_domain_features(tr))
dummy = trf.head(1000).copy()
tu, _ = k.compute_kfold_aggregates(trf, dummy, splits, ["Year", "Race", "Driver"], "unit")
ts, _ = k.compute_kfold_aggregates(trf, dummy, splits, ["Year", "Race", "Driver", "Stint"], "stint")
trf = pd.concat([trf, tu, ts], axis=1)

cats = [c for c in k.ALL_CAT_FEATURES if c in trf.columns and c != "IsOriginalData"]
high_card = ["Driver_Race", "Race_Compound_Stint", "Race_Year", "Driver_Compound", "Race_Compound"]
feat = [c for c in trf.columns if c not in ("id", "PitNextLap")]
for c in cats:
    trf[c] = trf[c].astype(str)

PARAMS = dict(
    iterations=800,
    learning_rate=0.05,
    depth=8,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    verbose=False,
    allow_writing_files=False,
)


def oof(use_feats, cat_feats):
    o = np.zeros(len(trf))
    for tri, vai in splits:
        m = CatBoostClassifier(**PARAMS)
        m.fit(trf.iloc[tri][use_feats], y[tri], cat_features=cat_feats, verbose=False)
        o[vai] = m.predict_proba(trf.iloc[vai][use_feats])[:, 1]
    return roc_auc_score(y, o)


numeric = [c for c in feat if c not in cats]
print(f"[A] numeric-only (no cats)            : {oof(numeric, []):.4f}")
print(f"[B] numeric + ALL cats (submit-02)    : {oof(feat, cats):.4f}   <- inflation?")
low_card = [c for c in cats if c not in high_card]
keep = numeric + low_card
print(f"[C] numeric + low-card cats only      : {oof(keep, low_card):.4f}")
print("--- iter=800 (submit-02 used 16000); LightGBM base+aggs=0.9318; LB=0.94296 ---")
