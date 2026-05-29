# ruff: noqa: E402
"""submit-07 — isotonic blend of Plan-2c CatBoost + RealMLP (architectural diversity).

submit-04 proved GBDT+GBDT does not blend (rho=0.958, +0.000pp). RealMLP is a
DIFFERENT architecture (feed-forward NN) -> expected lower correlation -> a blend
can lift AUC (the documented 699182 lever). This script does the blend OFFLINE on
prediction vectors only (tiny memory, no OOM risk — safe locally).

Inputs (each dir has oof_predictions.parquet [id,oof_pred,y] + test_predictions.parquet [id,test_pred]):
    --plan2c-dir   Plan-2c CatBoost (LB 0.95083, OOF 0.95676)
    --realmlp-dir  RealMLP kernel output

Methods compared on OOF AUC (all leak-free: single scalar weight / monotone maps):
    1. linear-raw      : w*cb + (1-w)*realmlp, grid w
    2. rank-average    : w*rank(cb) + (1-w)*rank(realmlp), grid w   (scale-free, AUC-native)
    3. isotonic-linear : isotonic-calibrate each (OOF->y), then linear blend, grid w
       (isotonic preserves per-model AUC exactly AND fixes per-year mean drift — 699182)

Picks the best method by OOF AUC, applies the SAME transform to test, writes:
    submissions/07_blend.csv          best blended submission
    submissions/07_blend_report.json  all methods' OOF AUC + corr + chosen w

Ship gate (per CLAUDE.md): submit to competition ONLY if blend OOF clearly beats
Plan-2c 0.95676 AND the gain is expected to transfer (corr low => real diversity).
LB is the only verdict. seed=42. CPU, vectors only.

Run:
    .venv/Scripts/python.exe scripts/run_07_isotonic_blend.py \
        --plan2c-dir DIR --realmlp-dir DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
SUBMISSIONS = REPO / "submissions"
ID, TARGET = "id", "PitNextLap"
EXPECTED_TEST_ROWS = 188_165


def _ranks(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(method="average").to_numpy() / len(x)


def lin(w, a, b):
    return w * a + (1 - w) * b


def _best_w(y, a, b, transform):
    """Grid-search w in [0,1] maximizing AUC of transform(w,a,b)."""
    ws = np.linspace(0.0, 1.0, 101)
    aucs = [roc_auc_score(y, transform(w, a, b)) for w in ws]
    i = int(np.argmax(aucs))
    return float(ws[i]), float(aucs[i])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan2c-dir", default="C:/Users/pierr/AppData/Local/Temp/plan2c_out")
    ap.add_argument("--realmlp-dir", default="C:/Users/pierr/AppData/Local/Temp/realmlp_out")
    args = ap.parse_args()
    cb_dir, mlp_dir = Path(args.plan2c_dir), Path(args.realmlp_dir)

    print("=" * 74)
    print("submit-07 — isotonic blend: Plan-2c CatBoost + RealMLP")
    print("=" * 74)

    cb_oof = pd.read_parquet(cb_dir / "oof_predictions.parquet")
    mlp_oof = pd.read_parquet(mlp_dir / "oof_predictions.parquet")
    cb_test = pd.read_parquet(cb_dir / "test_predictions.parquet")
    mlp_test = pd.read_parquet(mlp_dir / "test_predictions.parquet")

    # Align OOF by id
    m = (
        cb_oof[[ID, "oof_pred", "y"]]
        .rename(columns={"oof_pred": "cb"})
        .merge(mlp_oof[[ID, "oof_pred", "y"]].rename(columns={"oof_pred": "mlp", "y": "y2"}), on=ID)
    )
    assert (m["y"] == m["y2"]).all(), "y mismatch between the two OOF sets"
    assert len(m) == len(cb_oof) == len(mlp_oof), "OOF id misalignment"
    y = m["y"].to_numpy()
    cb, mlp = m["cb"].to_numpy(), m["mlp"].to_numpy()

    cb_auc = roc_auc_score(y, cb)
    mlp_auc = roc_auc_score(y, mlp)
    corr = float(np.corrcoef(cb, mlp)[0, 1])
    print(f"  CatBoost OOF AUC = {cb_auc:.5f}")
    print(f"  RealMLP  OOF AUC = {mlp_auc:.5f}")
    print(f"  corr(cb, mlp)    = {corr:.4f}  (lower = more diverse; submit-04 GBDT was 0.958)")

    # Align test by id (preserve cb_test order for submission)
    t = (
        cb_test[[ID, "test_pred"]]
        .rename(columns={"test_pred": "cb"})
        .merge(mlp_test[[ID, "test_pred"]].rename(columns={"test_pred": "mlp"}), on=ID)
    )
    assert len(t) == EXPECTED_TEST_ROWS, f"test rows {len(t)} != {EXPECTED_TEST_ROWS}"
    cb_t, mlp_t = t["cb"].to_numpy(), t["mlp"].to_numpy()

    # Isotonic calibrators fit on OOF (monotone -> AUC-preserving), applied to test.
    iso_cb = IsotonicRegression(out_of_bounds="clip").fit(cb, y)
    iso_mlp = IsotonicRegression(out_of_bounds="clip").fit(mlp, y)
    cb_cal, mlp_cal = iso_cb.predict(cb), iso_mlp.predict(mlp)
    cb_t_cal, mlp_t_cal = iso_cb.predict(cb_t), iso_mlp.predict(mlp_t)

    cb_r, mlp_r = _ranks(cb), _ranks(mlp)

    methods = {}
    w_lin, auc_lin = _best_w(y, cb, mlp, lin)
    methods["linear_raw"] = {"w": w_lin, "oof_auc": auc_lin}
    w_rank, auc_rank = _best_w(y, cb_r, mlp_r, lin)
    methods["rank_avg"] = {"w": w_rank, "oof_auc": auc_rank}
    w_iso, auc_iso = _best_w(y, cb_cal, mlp_cal, lin)
    methods["isotonic_linear"] = {"w": w_iso, "oof_auc": auc_iso}

    print("\n[methods OOF AUC]")
    for name, d in methods.items():
        print(f"  {name:16s} w={d['w']:.2f}  OOF AUC={d['oof_auc']:.5f}  (vs cb {cb_auc:.5f})")

    best_name = max(methods, key=lambda k: methods[k]["oof_auc"])
    best = methods[best_name]
    gain = best["oof_auc"] - cb_auc
    print(
        f"\n  BEST = {best_name} (w={best['w']:.2f}) OOF {best['oof_auc']:.5f} ({gain * 100:+.3f}pp vs CatBoost)"
    )

    # Build the test submission with the chosen method
    w = best["w"]
    if best_name == "linear_raw":
        test_blend = lin(w, cb_t, mlp_t)
    elif best_name == "rank_avg":
        test_blend = lin(w, _ranks(cb_t), _ranks(mlp_t))  # ranks in [0,1], valid probas
    else:
        test_blend = lin(w, cb_t_cal, mlp_t_cal)
    test_blend = np.clip(test_blend, 0.0, 1.0)
    assert not np.isnan(test_blend).any() and test_blend.shape == (EXPECTED_TEST_ROWS,)

    SUBMISSIONS.mkdir(exist_ok=True)
    pd.DataFrame({ID: t[ID].to_numpy(), TARGET: test_blend}).to_csv(
        SUBMISSIONS / "07_blend.csv", index=False
    )
    report = {
        "cb_oof_auc": cb_auc,
        "mlp_oof_auc": mlp_auc,
        "corr": corr,
        "methods": methods,
        "best_method": best_name,
        "best_w": w,
        "best_oof_auc": best["oof_auc"],
        "gain_pp_vs_catboost": gain * 100,
        "test_mean": float(test_blend.mean()),
    }
    (SUBMISSIONS / "07_blend_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"\n  wrote {SUBMISSIONS / '07_blend.csv'} (mean={test_blend.mean():.4f})")
    print(
        "  Ship gate: probe-submit ONLY if blend OOF clearly > 0.95676 AND corr indicates real diversity."
    )


if __name__ == "__main__":
    main()
