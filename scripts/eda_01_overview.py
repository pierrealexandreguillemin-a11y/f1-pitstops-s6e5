"""EDA #01 — Overview of train/test before submit-01 baseline.

Read-only: load train + test, dump structured findings to stdout (captured into
docs/eda/2026-05-27-overview.md). No model fitting, no submission.

Sections:
  1. shapes + dtypes
  2. missingness
  3. cardinality + value lists (small categoricals)
  4. Race x Year groups for GroupKFold (count, size dist, target rate per group)
  5. PitStop vs PitNextLap LEAK CHECK (most critical)
  6. numeric distributions (percentiles + outlier hints)
  7. train vs test drift (unseen values, mean/std comparison)
  8. target rate by key features (Compound, Stint, TyreLife buckets, RaceProgress)
  9. id contiguity sanity
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data import load_test, load_train  # noqa: E402

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda x: f"{x:.6f}")


def header(title: str) -> None:
    print(f"\n{'=' * 78}\n# {title}\n{'=' * 78}")


def main() -> None:
    tr = load_train()
    te = load_test()

    # === 1. shapes + dtypes ===
    header("1. SHAPES + DTYPES")
    print(f"train: {tr.shape}")
    print(f"test : {te.shape}")
    print("\ntrain dtypes:")
    print(tr.dtypes.to_string())
    print("\ntest dtypes:")
    print(te.dtypes.to_string())

    # === 2. missingness ===
    header("2. MISSINGNESS")
    miss_tr = tr.isna().sum()
    miss_te = te.isna().sum()
    miss = pd.DataFrame(
        {
            "train_na": miss_tr,
            "train_pct": (miss_tr / len(tr) * 100).round(3),
            "test_na": miss_te.reindex(miss_tr.index, fill_value=0),
            "test_pct": (miss_te.reindex(miss_tr.index, fill_value=0) / len(te) * 100).round(3),
        }
    )
    print(miss.to_string())

    # === 3. cardinality ===
    header("3. CARDINALITY (object + low-card numeric)")
    cards = []
    for c in tr.columns:
        cards.append(
            {
                "col": c,
                "dtype": str(tr[c].dtype),
                "nunique_train": tr[c].nunique(dropna=False),
                "nunique_test": te[c].nunique(dropna=False) if c in te.columns else None,
            }
        )
    print(pd.DataFrame(cards).to_string(index=False))

    print("\n-- Compound values (train):")
    print(tr["Compound"].value_counts(dropna=False).to_string())
    print("\n-- Compound values (test):")
    print(te["Compound"].value_counts(dropna=False).to_string())
    print("\n-- Year values (train):")
    print(tr["Year"].value_counts(dropna=False).sort_index().to_string())
    print("\n-- Year values (test):")
    print(te["Year"].value_counts(dropna=False).sort_index().to_string())
    print(f"\n-- Race nunique train: {tr['Race'].nunique()}  | test: {te['Race'].nunique()}")
    print(f"-- Driver nunique train: {tr['Driver'].nunique()}  | test: {te['Driver'].nunique()}")

    # === 4. Race x Year groups for GroupKFold ===
    header("4. GROUPS Race x Year (for GroupKFold)")
    g_tr = tr["Race"].astype(str) + "-" + tr["Year"].astype(str)
    g_te = te["Race"].astype(str) + "-" + te["Year"].astype(str)
    print(f"n_groups train: {g_tr.nunique()}")
    print(f"n_groups test : {g_te.nunique()}")
    print(f"overlap (groups in both train AND test): {len(set(g_tr) & set(g_te))}")
    print(f"only in test (unseen by train): {len(set(g_te) - set(g_tr))}")

    sizes = g_tr.value_counts()
    print(
        f"\ngroup size train — min:{sizes.min()} p25:{int(sizes.quantile(0.25))} "
        f"median:{int(sizes.median())} p75:{int(sizes.quantile(0.75))} max:{sizes.max()} "
        f"mean:{sizes.mean():.1f}"
    )

    rate_per_group = tr.groupby(g_tr)["PitNextLap"].mean()
    print(
        f"target rate per group — min:{rate_per_group.min():.4f} "
        f"p25:{rate_per_group.quantile(0.25):.4f} "
        f"median:{rate_per_group.median():.4f} "
        f"p75:{rate_per_group.quantile(0.75):.4f} "
        f"max:{rate_per_group.max():.4f}"
    )
    print(f"groups with 0% positives: {(rate_per_group == 0).sum()}")
    print(f"groups with 100% positives: {(rate_per_group == 1).sum()}")

    # === 5. PITSTOP vs PITNEXTLAP — LEAK CHECK ===
    header("5. LEAK CHECK — PitStop vs PitNextLap")
    print("\nCross-tab (rows=PitStop this lap, cols=PitNextLap):")
    print(pd.crosstab(tr["PitStop"], tr["PitNextLap"], margins=True))
    print("\nP(PitNextLap=1 | PitStop=this_lap):")
    print(tr.groupby("PitStop")["PitNextLap"].agg(["mean", "count"]).to_string())

    # Same driver/race: does PitNextLap[lap=L] == PitStop[lap=L+1] ?
    # Sort by Race, Year, Driver, LapNumber and shift.
    sub = tr.sort_values(["Race", "Year", "Driver", "LapNumber"]).copy()
    sub["PitStop_nextlap"] = sub.groupby(["Race", "Year", "Driver"])["PitStop"].shift(-1)
    valid = sub.dropna(subset=["PitStop_nextlap"])
    match = (valid["PitNextLap"] == valid["PitStop_nextlap"]).mean()
    print(
        f"\nP(PitNextLap == PitStop[next lap, same driver/race]): {match:.6f}  "
        f"(over {len(valid):,} rows with a valid next lap)"
    )

    # === 6. numeric distributions ===
    header("6. NUMERIC DISTRIBUTIONS (train)")
    num = tr.select_dtypes(include=[np.number]).drop(columns=["id", "PitNextLap", "PitStop"])
    desc = num.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T
    print(desc.to_string())

    # === 7. train vs test drift ===
    header("7. TRAIN vs TEST DRIFT")
    print("Unseen categorical values in test:")
    for c in ["Driver", "Compound", "Race", "Year"]:
        unseen = set(te[c].dropna().unique()) - set(tr[c].dropna().unique())
        print(f"  {c}: {len(unseen)} unseen  (sample: {list(unseen)[:5]})")

    print("\nNumeric mean/std comparison:")
    common = [c for c in num.columns if c in te.columns and pd.api.types.is_numeric_dtype(te[c])]
    drift = pd.DataFrame(
        {
            "train_mean": tr[common].mean(),
            "test_mean": te[common].mean(),
            "train_std": tr[common].std(),
            "test_std": te[common].std(),
        }
    )
    drift["mean_delta_pct"] = (
        (drift["test_mean"] - drift["train_mean"]).abs()
        / drift["train_std"].replace(0, np.nan)
        * 100
    ).round(2)
    print(drift.to_string())

    # === 8. target rate by key features ===
    header("8. TARGET RATE BY KEY FEATURES")

    print("\n-- by Compound:")
    print(
        tr.groupby("Compound")["PitNextLap"]
        .agg(["mean", "count"])
        .sort_values("count", ascending=False)
        .to_string()
    )

    print("\n-- by Stint:")
    print(tr.groupby("Stint")["PitNextLap"].agg(["mean", "count"]).head(10).to_string())

    print("\n-- by TyreLife bucket (quintiles):")
    tl_buckets = pd.qcut(tr["TyreLife"], 5, duplicates="drop")
    print(tr.groupby(tl_buckets, observed=True)["PitNextLap"].agg(["mean", "count"]).to_string())

    print("\n-- by RaceProgress bucket (quintiles):")
    rp_buckets = pd.qcut(tr["RaceProgress"], 5, duplicates="drop")
    print(tr.groupby(rp_buckets, observed=True)["PitNextLap"].agg(["mean", "count"]).to_string())

    print("\n-- by Position (top 10 positions):")
    print(tr.groupby("Position")["PitNextLap"].agg(["mean", "count"]).head(10).to_string())

    print("\n-- by Year:")
    print(tr.groupby("Year")["PitNextLap"].agg(["mean", "count"]).sort_index().to_string())

    # === 9. id sanity ===
    header("9. ID CONTIGUITY / OVERLAP")
    print(
        f"train id min/max: {tr['id'].min()} / {tr['id'].max()}  unique: {tr['id'].nunique() == len(tr)}"
    )
    print(
        f"test  id min/max: {te['id'].min()} / {te['id'].max()}  unique: {te['id'].nunique() == len(te)}"
    )
    print(f"id overlap train/test: {len(set(tr['id']) & set(te['id']))}")


if __name__ == "__main__":
    main()
