# f1-best-catboost-solution-0-95259

## [cell 0 MD]
# Pit Stop Prediction OOP Solution: Seed Ensemble + Meta Stacking

This notebook builds a readable object-oriented CatBoost pipeline for the F1 pit stop prediction task. It combines the competition data with the additional F1 strategy dataset, creates domain, categorical, frequency, group-statistic, digit, and signature features, then trains a strong CatBoost baseline. The main production signal is still the stable two-seed CatBoost ensemble built from seeds `42` and `777`.

The latest research direction is more selective than the earlier experiments. Extra random seeds and direct `shallow`/`reg` CatBoost variant injections did not improve the final public blend, so the notebook now avoids creating many near-duplicate submissions. Instead, it saves only compact blend inputs and adds a second-level Meta CatBoost stacker trained on out-of-fold predictions from several feature-family views.

The final goal of this notebook is to produce one clean internal signal, `outputs/stacking/meta.csv`, that can be tested in the external blender as a tiny addition to the current best public blend. This keeps the modeling pipeline useful without turning the output folder into a large set of speculative submissions.


## [cell 1 CODE]
```python
"""Import libraries, define settings, and make the notebook reproducible."""

import os
import gc
import random
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from html import escape
from IPython.display import display, HTML

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)
pd.set_option("display.float_format", lambda x: f"{x:.6f}")


@dataclass
class Config:
    seed: int = 42
    target: str = "PitNextLap"
    id_col: str = "id"
    validation_mode: str = "single_split"
    run_validation: bool = False
    fixed_final_iterations: int | None = 8140
    valid_size: float = 0.20
    n_folds: int = 5
    ensemble_seeds: tuple[int, ...] = (42, 777)
    stacking_n_folds: int = 3
    stacking_base_iterations: int = 5000
    stacking_meta_iterations: int = 3000
    require_original_data: bool = True
    comp_paths: list[str] = field(
        default_factory=lambda: [
            "/kaggle/input/competitions/playground-series-s6e5",
            "/kaggle/input/playground-series-s6e5",
        ]
    )
    original_paths: list[str] = field(
        default_factory=lambda: [
            "/kaggle/input/f1-strategy-dataset-pit-stop-prediction/f1_strategy_dataset_v4.csv",
            "/kaggle/input/datasets/aadigupta1601/f1-strategy-dataset-pit-stop-prediction/f1_strategy_dataset_v4.csv",
            "/kaggle/input/f1-strategy-dataset-pit-stop-prediction/f1_strategy_dataset_v4.csv",
        ]
    )
    numeric_features: list[str] = field(
        default_factory=lambda: [
            "Year",
            "PitStop",
            "LapNumber",
            "Stint",
            "TyreLife",
            "Position",
            "LapTime (s)",
            "LapTime_Delta",
            "Cumulative_Degradation",
            "RaceProgress",
            "Position_Change",
        ]
    )
    categorical_features: list[str] = field(default_factory=lambda: ["Driver", "Compound", "Race"])
    base_cat_cols: list[str] = field(default_factory=lambda: ["Driver", "Compound", "Race"])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def first_existing_path(paths: list[str]) -> str:
    for path in paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No valid path found from: {paths}")


cfg = Config()
seed_everything(cfg.seed)



def _fmt_value(value):
    if pd.isna(value):
        return "-"
    if isinstance(value, (np.integer, int)):
        return f"{int(value):,}"
    if isinstance(value, (np.floating, float)):
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def show_title(title, subtitle=None):
    subtitle_html = f'<div style="color:#6b7280;font-size:13px;margin-top:3px;">{escape(str(subtitle))}</div>' if subtitle else ""
    display(HTML(
        f"""
        <div style="margin:18px 0 10px 0;padding-bottom:8px;border-bottom:1px solid #e5e7eb;">
          <div style="font-size:18px;font-weight:700;color:#111827;">{escape(str(title))}</div>
          {subtitle_html}
        </div>
        """
    ))


def show_metric_cards(title, data, columns=4):
    if isinstance(data, pd.DataFrame):
        data = data.iloc[0].to_dict() if len(data) else {}
    cards = []
    for key, value in data.items():
        cards.append(
            f"""
            <div style="border:1px solid #e5e7eb;border-radius:8px;padding:10px 12px;background:#ffffff;">
              <div style="font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#6b7280;">{escape(str(key).replace('_', ' '))}</div>
              <div style="font-size:18px;font-weight:700;color:#111827;margin-top:4px;">{escape(_fmt_value(value))}</div>
            </div>
            """
        )
    display(HTML(
        f"""
        <div style="margin:12px 0;">
          <div style="font-size:15px;font-weight:700;color:#111827;margin-bottom:8px;">{escape(str(title))}</div>
          <div style="display:grid;grid-template-columns:repeat({columns},minmax(0,1fr));gap:8px;">
            {''.join(cards)}
          </div>
        </div>
        """
    ))


def show_record_cards(title, df, max_rows=8, columns=2):
    view = df.head(max_rows).copy()
    records = []
    for _, row in view.iterrows():
        fields = []
        for key, value in row.items():
            fields.append(
                f"<div><span style='color:#6b7280'>{escape(str(key).replace('_', ' '))}</span>: "
                f"<strong style='color:#111827'>{escape(_fmt_value(value))}</strong></div>"
            )
        records.append(
            f"""
            <div style="border:1px solid #e5e7eb;border-radius:8px;padding:10px 12px;background:#ffffff;">
              <div style="font-size:13px;line-height:1.65;">{''.join(fields)}</div>
            </div>
            """
        )
    more = "" if len(df) <= max_rows else f"<div style='color:#6b7280;font-size:12px;margin-top:6px;'>Showing {max_rows} of {len(df)} records</div>"
    display(HTML(
        f"""
        <div style="margin:12px 0;">
          <div style="font-size:15px;font-weight:700;color:#111827;margin-bottom:8px;">{escape(str(title))}</div>
          <div style="display:grid;grid-template-columns:repeat({columns},minmax(0,1fr));gap:8px;">
            {''.join(records)}
          </div>
          {more}
        </div>
        """
    ))


def show_feature_list(title, values, max_items=40):
    shown = list(values)[:max_items]
    items = ''.join(
        f"<span style='display:inline-block;border:1px solid #e5e7eb;border-radius:999px;padding:4px 8px;margin:3px;background:#ffffff;font-size:12px;color:#111827;'>{escape(str(v))}</span>"
        for v in shown
    )
    more = "" if len(values) <= max_items else f"<div style='color:#6b7280;font-size:12px;margin-top:6px;'>Showing {max_items} of {len(values)} features</div>"
    display(HTML(
        f"""
        <div style="margin:12px 0;">
          <div style="font-size:15px;font-weight:700;color:#111827;margin-bottom:6px;">{escape(str(title))}</div>
          <div>{items}</div>
          {more}
        </div>
        """
    ))


def show_ranked_features(title, df, max_rows=25):
    view = df.head(max_rows).reset_index(drop=True)
    if len(view) == 0:
        return
    max_imp = max(float(view["importance"].max()), 1e-12)
    rows = []
    for idx, row in view.iterrows():
        importance = float(row["importance"])
        width = max(2, min(100, importance / max_imp * 100))
        rows.append(
            f"""
            <div style="display:grid;grid-template-columns:34px 1fr 82px;gap:10px;align-items:center;margin:6px 0;">
              <div style="color:#6b7280;font-size:12px;text-align:right;">{idx + 1}</div>
              <div>
                <div style="font-size:13px;color:#111827;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{escape(str(row['feature']))}</div>
                <div style="height:6px;background:#eef2f7;border-radius:999px;margin-top:4px;">
                  <div style="width:{width:.1f}%;height:6px;background:#374151;border-radius:999px;"></div>
                </div>
              </div>
              <div style="font-size:12px;color:#111827;text-align:right;font-weight:700;">{importance:.6f}</div>
            </div>
            """
        )
    display(HTML(
        f"""
        <div style="margin:12px 0;border:1px solid #e5e7eb;border-radius:8px;padding:12px;background:#ffffff;">
          <div style="font-size:15px;font-weight:700;color:#111827;margin-bottom:8px;">{escape(str(title))}</div>
          {''.join(rows)}
        </div>
        """
    ))
```

## [cell 2 MD]
## Data Loading and Checks

## [cell 3 CODE]
```python
"""Load train, test, submission, and optional original data, then show compact dataset checks."""

class DataLoader:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.comp_path = None
        self.original_path = None

    def load(self):
        self.comp_path = first_existing_path(self.cfg.comp_paths)
        train = pd.read_csv(os.path.join(self.comp_path, "train.csv"))
        test = pd.read_csv(os.path.join(self.comp_path, "test.csv"))
        sample_submission = pd.read_csv(os.path.join(self.comp_path, "sample_submission.csv"))
        original = self._load_original()
        if original is None and self.cfg.require_original_data:
            raise FileNotFoundError(
                "Original dataset was not found. Add Kaggle input: "
                "F1 Strategy Dataset | Pit Stop Prediction, then rerun the notebook."
            )
        return train, test, sample_submission, original

    def _load_original(self):
        for path in self.cfg.original_paths:
            if os.path.exists(path):
                self.original_path = path
                return pd.read_csv(path)
        return None

    def path_report(self):
        return pd.DataFrame(
            [
                {"source": "competition", "path": self.comp_path, "loaded": self.comp_path is not None},
                {"source": "original", "path": self.original_path, "loaded": self.original_path is not None},
            ]
        )

    def dataset_overview(self, train, test, original):
        rows = [
            self._one_dataset_row("train", train),
            self._one_dataset_row("test", test),
        ]
        if original is not None:
            rows.append(self._one_dataset_row("original", original))
        return pd.DataFrame(rows)

    def _one_dataset_row(self, name, df):
        target_rate = np.nan
        if self.cfg.target in df.columns:
            target_rate = df[self.cfg.target].mean()
        id_duplicates = np.nan
        if self.cfg.id_col in df.columns:
            id_duplicates = df[self.cfg.id_col].duplicated().sum()
        return {
            "dataset": name,
            "rows": len(df),
            "columns": df.shape[1],
            "target_rate": target_rate,
            "missing_values": int(df.isna().sum().sum()),
            "duplicate_ids": id_duplicates,
            "duplicate_rows": int(df.duplicated().sum()),
        }

    def schema_summary(self, df, name):
        return pd.DataFrame(
            {
                "dataset": name,
                "column": df.columns,
                "dtype": [str(df[col].dtype) for col in df.columns],
                "missing": [int(df[col].isna().sum()) for col in df.columns],
                "missing_pct": [100 * df[col].isna().mean() for col in df.columns],
                "unique": [df[col].nunique() for col in df.columns],
            }
        ).sort_values(["missing_pct", "unique"], ascending=[False, False])


loader = DataLoader(cfg)
train_raw, test_raw, sample_submission, original_raw = loader.load()

show_title("Data loading", "Competition data and optional original dataset")
show_record_cards("Input paths", loader.path_report(), max_rows=4, columns=1)
show_record_cards("Dataset overview", loader.dataset_overview(train_raw, test_raw, original_raw), max_rows=3, columns=3)
show_record_cards("Train schema snapshot", loader.schema_summary(train_raw, "train").head(8), max_rows=8, columns=2)

if original_raw is not None:
    show_record_cards("Original schema snapshot", loader.schema_summary(original_raw, "original").head(8), max_rows=8, columns=2)

gc.collect()
```

## [cell 4 MD]
## Quick Drift View

## [cell 5 CODE]
```python
"""Compare important train and test columns with small drift and coverage tables."""

class DataInspector:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def numeric_drift(self, train_df, test_df):
        rows = []
        for col in self.cfg.numeric_features:
            if col not in train_df.columns or col not in test_df.columns:
                continue
            train_values = train_df[col].astype(float)
            test_values = test_df[col].astype(float)
            pooled_std = np.sqrt((train_values.var() + test_values.var()) / 2.0)
            smd = 0.0 if pooled_std == 0 else (test_values.mean() - train_values.mean()) / pooled_std
            rows.append(
                {
                    "feature": col,
                    "train_mean": train_values.mean(),
                    "test_mean": test_values.mean(),
                    "smd": smd,
                }
            )
        return pd.DataFrame(rows).sort_values("smd", key=np.abs, ascending=False)

    def categorical_coverage(self, train_df, test_df):
        rows = []
        for col in self.cfg.categorical_features:
            if col not in train_df.columns or col not in test_df.columns:
                continue
            train_values = set(train_df[col].astype("string").dropna().unique())
            test_values = set(test_df[col].astype("string").dropna().unique())
            rows.append(
                {
                    "feature": col,
                    "train_unique": len(train_values),
                    "test_unique": len(test_values),
                    "unseen_in_test": len(test_values - train_values),
                }
            )
        return pd.DataFrame(rows)


inspector = DataInspector(cfg)
show_title("Quick drift view", "Train/test numeric drift and categorical coverage")
show_record_cards("Numeric drift", inspector.numeric_drift(train_raw, test_raw), max_rows=11, columns=2)
show_record_cards("Categorical coverage", inspector.categorical_coverage(train_raw, test_raw), max_rows=3, columns=3)
```

## [cell 6 MD]
## Feature Builder: Signature, Cross, Frequency, and Group Statistics

## [cell 7 CODE]
```python
"""Create domain, signature, cross-categorical, frequency, and group-stat features for CatBoost."""

class FeatureEngineer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cat_cols: list[str] = []
        self.num_cols: list[str] = []
        self.feature_cols: list[str] = []
        self.frequency_cols: list[str] = []
        self.group_stat_features: list[str] = []
        self.digit_source_cols: list[str] = []
        self.digit_features: list[str] = []
        self.signature_features: list[str] = []
        self.string_precision_features: list[str] = []

    @staticmethod
    def safe_div(a, b, eps=1e-6):
        return a / (b + eps)

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        eps = 1e-6

        for col in self.cfg.base_cat_cols:
            if col in out.columns:
                out[col] = out[col].astype("string").fillna("__MISSING__").astype(str)

        def has(cols):
            return set(cols).issubset(out.columns)

        if has(["LapNumber", "RaceProgress"]):
            race_progress = out["RaceProgress"].clip(lower=eps)
            est_total = self.safe_div(out["LapNumber"], race_progress, eps).replace([np.inf, -np.inf], np.nan)
            out["EstimatedTotalLaps"] = est_total.clip(1, 120)
            out["LapsRemaining"] = (out["EstimatedTotalLaps"] - out["LapNumber"]).clip(lower=0)
            out["RemainingRaceProgress"] = 1.0 - out["RaceProgress"]
            out["LapProgress_x_LapNumber"] = out["LapNumber"] * out["RaceProgress"]
            out["Early_Race"] = (out["RaceProgress"] <= 0.25).astype(np.int8)
            out["Mid_Race"] = ((out["RaceProgress"] > 0.25) & (out["RaceProgress"] <= 0.65)).astype(np.int8)
            out["Late_Race"] = (out["RaceProgress"] > 0.65).astype(np.int8)
            out["RacePhase"] = pd.cut(
                out["RaceProgress"],
                bins=[-np.inf, 0.20, 0.40, 0.60, 0.80, np.inf],
                labels=["P1", "P2", "P3", "P4", "P5"],
            ).astype(str)
            out["LapBin"] = pd.cut(
                out["LapNumber"],
                bins=[-np.inf, 5, 10, 20, 35, 50, np.inf],
                labels=["L_000_005", "L_006_010", "L_011_020", "L_021_035", "L_036_050", "L_051_plus"],
            ).astype(str)

        if has(["TyreLife", "LapNumber"]):
            out["TyreAgeRatio"] = self.safe_div(out["TyreLife"], out["LapNumber"].clip(lower=1), eps)
            out["LapPerTyreLife"] = self.safe_div(out["LapNumber"], out["TyreLife"] + 1, eps)
            out["TyreLifeMinusLap"] = out["TyreLife"] - out["LapNumber"]
            out["LapMinusTyreLife"] = out["LapNumber"] - out["TyreLife"]

        if has(["TyreLife", "EstimatedTotalLaps"]):
            out["TyreAgeVsRace"] = self.safe_div(out["TyreLife"], out["EstimatedTotalLaps"].clip(lower=1), eps)

        if has(["TyreLife", "RaceProgress"]):
            out["PitWindowPressure"] = out["TyreLife"] * out["RaceProgress"]
            out["TyreLife_x_RaceProgress"] = out["TyreLife"] * out["RaceProgress"]

        if has(["TyreLife", "LapsRemaining"]):
            out["TyreLife_to_LapsRemaining"] = self.safe_div(out["TyreLife"], out["LapsRemaining"] + 1, eps)
            out["LapsRemaining_to_TyreLife"] = self.safe_div(out["LapsRemaining"], out["TyreLife"] + 1, eps)

        if has(["Stint", "TyreLife"]):
            out["StintPressure"] = out["Stint"] * out["TyreLife"]
            out["TyreLife_x_Stint"] = out["TyreLife"] * out["Stint"]
            out["Is_First_Stint"] = (out["Stint"] == 1).astype(np.int8)
            out["Is_Late_Stint"] = (out["Stint"] >= 3).astype(np.int8)

        if has(["Stint", "LapNumber"]):
            out["Stint_x_LapNumber"] = out["Stint"] * out["LapNumber"]

        if "TyreLife" in out.columns:
            out["TyreLifeBin"] = pd.cut(
                out["TyreLife"],
                bins=[-np.inf, 3, 7, 12, 20, 30, np.inf],
                labels=["T_000_003", "T_004_007", "T_008_012", "T_013_020", "T_021_030", "T_031_plus"],
            ).astype(str)

        if "Position" in out.columns:
            out["PositionBin"] = pd.cut(
                out["Position"],
                bins=[-np.inf, 3, 8, 14, np.inf],
                labels=["front", "upper_mid", "lower_mid", "back"],
            ).astype(str)

        if has(["Cumulative_Degradation", "LapNumber"]):
            out["DegPerRaceLap"] = self.safe_div(out["Cumulative_Degradation"], out["LapNumber"].clip(lower=1), eps)

        if has(["Cumulative_Degradation", "TyreLife"]):
            out["DegPerTyreLap"] = self.safe_div(out["Cumulative_Degradation"], out["TyreLife"].clip(lower=1), eps)
            out["AbsDegPerTyreLap"] = self.safe_div(out["Cumulative_Degradation"].abs(), out["TyreLife"].clip(lower=1), eps)

        if "Cumulative_Degradation" in out.columns:
            out["Abs_Cumulative_Degradation"] = out["Cumulative_Degradation"].abs()
            out["Positive_Degradation"] = (out["Cumulative_Degradation"] > 0).astype(np.int8)

        if "LapTime_Delta" in out.columns:
            out["DeltaAbs"] = out["LapTime_Delta"].abs()
            out["LapTimeDeltaPositive"] = (out["LapTime_Delta"] > 0).astype(np.int8)
            out["LapTimeDeltaNegative"] = (out["LapTime_Delta"] < 0).astype(np.int8)

        if has(["LapTime_Delta", "TyreLife"]):
            out["DeltaPerTyreLap"] = self.safe_div(out["LapTime_Delta"], out["TyreLife"].clip(lower=1), eps)
            out["AbsDeltaPerTyreLap"] = out["DeltaPerTyreLap"].abs()

        if "Position_Change" in out.columns:
            out["Abs_Position_Change"] = out["Position_Change"].abs()
            out["Gained_Position"] = (out["Position_Change"] > 0).astype(np.int8)
            out["Lost_Position"] = (out["Position_Change"] < 0).astype(np.int8)

        if has(["Position", "RaceProgress"]):
            out["PositionPressure"] = out["Position"] * out["RaceProgress"]

        self._add_cross(out, "Race_Year", ["Race", "Year"])
        self._add_cross(out, "Compound_Stint", ["Compound", "Stint"])
        self._add_cross(out, "Driver_Race", ["Driver", "Race"])
        self._add_cross(out, "Driver_Compound", ["Driver", "Compound"])
        self._add_cross(out, "Race_Compound", ["Race", "Compound"])
        self._add_cross(out, "Race_Compound_Stint", ["Race", "Compound", "Stint"])
        self._add_cross(out, "Compound_RacePhase", ["Compound", "RacePhase"])
        self._add_cross(out, "Compound_TyreLifeBin", ["Compound", "TyreLifeBin"])
        self._add_cross(out, "RacePhase_TyreLifeBin", ["RacePhase", "TyreLifeBin"])

        out = out.replace([np.inf, -np.inf], np.nan)
        for col in out.select_dtypes(include=["float64"]).columns:
            out[col] = out[col].astype(np.float32)
        return out

    @staticmethod
    def _add_cross(df, name, cols):
        if set(cols).issubset(df.columns):
            value = df[cols[0]].astype(str)
            for col in cols[1:]:
                value = value + "_" + df[col].astype(str)
            df[name] = value

    def _add_frequency_features(self, train, test, original):
        frames = [train, test] + ([original] if original is not None else [])
        candidate_cols = [
            "Driver",
            "Race",
            "Compound",
            "Race_Year",
            "Compound_Stint",
            "Driver_Race",
            "Driver_Compound",
            "Race_Compound",
            "Race_Compound_Stint",
            "Compound_RacePhase",
            "Compound_TyreLifeBin",
            "RacePhase_TyreLifeBin",
            "LapBin",
            "TyreLifeBin",
            "PositionBin",
        ]
        self.frequency_cols = [col for col in candidate_cols if all(col in frame.columns for frame in frames)]

        total_rows = sum(len(frame) for frame in frames)
        for col in self.frequency_cols:
            values = pd.concat([frame[col].astype("string") for frame in frames], axis=0).fillna("__MISSING__")
            counts = values.value_counts(dropna=False)
            count_col = f"{col}_count"
            freq_col = f"{col}_freq"
            for frame in frames:
                keys = frame[col].astype("string").fillna("__MISSING__")
                frame[count_col] = keys.map(counts).fillna(0).astype(np.int32)
                frame[freq_col] = (frame[count_col] / total_rows).astype(np.float32)
        return train, test, original

    def _add_group_statistics(self, train, test, original):
        frames = [train, test] + ([original] if original is not None else [])
        group_cols = ["Race_Year", "Race_Compound_Stint", "Driver_Race", "Compound_Stint"]
        value_cols = ["LapTime_Delta", "Position_Change", "RaceProgress", "TyreLife"]
        self.group_stat_features = []

        combined = pd.concat(
            [frame[[col for col in set(group_cols + value_cols) if col in frame.columns]].copy() for frame in frames],
            axis=0,
            ignore_index=True,
        )

        for group_col in group_cols:
            if group_col not in combined.columns:
                continue
            for value_col in value_cols:
                if value_col not in combined.columns:
                    continue

                stats = combined.groupby(group_col, dropna=False)[value_col].agg(["mean", "std"])
                mean_col = f"{value_col}_mean_by_{group_col}"
                std_col = f"{value_col}_std_by_{group_col}"
                diff_col = f"{value_col}_diff_mean_by_{group_col}"
                self.group_stat_features.extend([mean_col, std_col, diff_col])

                for frame in frames:
                    keys = frame[group_col]
                    frame[mean_col] = keys.map(stats["mean"]).astype(np.float32)
                    frame[std_col] = keys.map(stats["std"]).fillna(0).astype(np.float32)
                    frame[diff_col] = (frame[value_col] - frame[mean_col]).astype(np.float32)

        return train, test, original

    def _get_digit_source_cols(self, train, test):
        candidates = [
            "Year",
            "PitStop",
            "LapNumber",
            "Stint",
            "TyreLife",
            "Position",
            "LapTime (s)",
            "LapTime_Delta",
            "Cumulative_Degradation",
            "RaceProgress",
            "Position_Change",
            "EstimatedTotalLaps",
            "LapsRemaining",
            "TyreAgeRatio",
            "DegPerTyreLap",
            "DegPerRaceLap",
            "DeltaPerTyreLap",
            "DeltaAbs",
            "PositionPressure",
            "StintPressure",
            "PitWindowPressure",
            "LapMinusTyreLife",
        ]
        return [col for col in candidates if col in train.columns and col in test.columns]

    def _add_digit_features(self, df, numeric_cols, int_digit_limit=3, decimal_digit_limit=2):
        out = df.copy()
        for col in numeric_cols:
            if col not in out.columns:
                continue
            values = out[col].fillna(0).astype(float).abs()
            for i in range(int_digit_limit):
                new_col = f"{col}_int_digit_{i + 1}"
                out[new_col] = ((values // (10 ** i)) % 10).astype(np.int8)
                if new_col not in self.digit_features:
                    self.digit_features.append(new_col)
            if pd.api.types.is_float_dtype(out[col]):
                for i in range(1, decimal_digit_limit + 1):
                    new_col = f"{col}_dec_digit_{i}"
                    out[new_col] = ((values * (10 ** i)).round().astype(int) % 10).astype(np.int8)
                    if new_col not in self.digit_features:
                        self.digit_features.append(new_col)
        return out

    def _add_float_signature_features(self, df):
        out = df.copy()
        selected = [
            "RaceProgress",
            "LapTime (s)",
            "LapTime_Delta",
            "Cumulative_Degradation",
            "TyreAgeRatio",
            "DegPerTyreLap",
            "DegPerRaceLap",
            "DeltaPerTyreLap",
            "DeltaAbs",
            "PitWindowPressure",
            "EstimatedTotalLaps",
            "LapsRemaining",
            "LapMinusTyreLife",
        ]
        for col in selected:
            if col not in out.columns:
                continue
            scaled = (out[col].fillna(0).astype(float) * 100).round().astype(int).abs()
            for i in range(5):
                new_col = f"{col}_sig_{i + 1}"
                digit = ((scaled // (10 ** i)) % 10).astype(np.int8)
                if digit.nunique() > 1:
                    out[new_col] = digit.astype(str)
                    if new_col not in self.signature_features:
                        self.signature_features.append(new_col)
        return out

    def _add_string_precision_features(self, df):
        out = df.copy()
        specs = {
            "RaceProgress": ("RaceProgress_str", 4),
            "EstimatedTotalLaps": ("EstimatedTotalLaps_str", 1),
            "TyreAgeRatio": ("TyreAgeRatio_str", 3),
        }
        for source_col, (new_col, precision) in specs.items():
            if source_col in out.columns:
                out[new_col] = out[source_col].round(precision).astype(str)
                if new_col not in self.string_precision_features:
                    self.string_precision_features.append(new_col)
        return out

    def transform_all(self, train_raw, test_raw, original_raw=None):
        train = train_raw.copy()
        test = test_raw.copy()
        original = original_raw.copy() if original_raw is not None and self.cfg.target in original_raw.columns else None

        train["IsOriginalData"] = 0
        test["IsOriginalData"] = 0
        if original is not None:
            original["IsOriginalData"] = 1
            original = original.drop(columns=["Normalized_TyreLife"], errors="ignore")

        train = self.add_features(train)
        test = self.add_features(test)
        original = self.add_features(original) if original is not None else None

        self.digit_source_cols = self._get_digit_source_cols(train, test)
        self.digit_features = []
        self.signature_features = []
        self.string_precision_features = []

        train = self._add_digit_features(train, self.digit_source_cols)
        test = self._add_digit_features(test, self.digit_source_cols)
        original = self._add_digit_features(original, self.digit_source_cols) if original is not None else None

        train = self._add_float_signature_features(train)
        test = self._add_float_signature_features(test)
        original = self._add_float_signature_features(original) if original is not None else None

        train = self._add_string_precision_features(train)
        test = self._add_string_precision_features(test)
        original = self._add_string_precision_features(original) if original is not None else None

        train, test, original = self._add_frequency_features(train, test, original)
        train, test, original = self._add_group_statistics(train, test, original)
        train, test, original = self._align_columns(train, test, original)
        train, test, original = self._fill_missing(train, test, original)
        return train, test, original

    def _align_columns(self, train, test, original):
        exclude_cols = [self.cfg.id_col, self.cfg.target]
        self.feature_cols = [col for col in train.columns if col in test.columns and col not in exclude_cols]
        train = train[self.feature_cols + [self.cfg.target]]
        test = test[self.feature_cols]

        if original is not None:
            for col in self.feature_cols:
                if col not in original.columns:
                    original[col] = np.nan
            original = original[self.feature_cols + [self.cfg.target]]
        return train, test, original

    def _fill_missing(self, train, test, original):
        frames = [train, test]
        if original is not None:
            frames.append(original)

        from pandas.api.types import is_categorical_dtype, is_object_dtype, is_string_dtype

        def is_categorical_like(series):
            return (
                is_object_dtype(series.dtype)
                or is_categorical_dtype(series.dtype)
                or is_string_dtype(series.dtype)
            )

        self.cat_cols = []
        for col in self.feature_cols:
            if any(is_categorical_like(frame[col]) for frame in frames if col in frame.columns):
                self.cat_cols.append(col)
        self.num_cols = [col for col in self.feature_cols if col not in self.cat_cols]

        for col in self.cat_cols:
            values = pd.concat([frame[col].astype("string") for frame in frames if col in frame.columns], axis=0)
            mode_value = values.mode().iloc[0] if len(values.mode()) else "__MISSING__"
            for frame in frames:
                if col in frame.columns:
                    frame[col] = frame[col].astype("string").fillna(mode_value).astype(str)

        for col in self.num_cols:
            values = pd.concat([frame[col] for frame in frames if col in frame.columns], axis=0)
            fill_value = values.replace([np.inf, -np.inf], np.nan).median()
            for frame in frames:
                if col in frame.columns:
                    frame[col] = frame[col].replace([np.inf, -np.inf], np.nan).fillna(fill_value)
                    if frame[col].dtype == "float64":
                        frame[col] = frame[col].astype(np.float32)
        return train, test, original

    def feature_report(self, train, test, original):
        rows = [
            {"dataset": "train", "rows": len(train), "columns": train.shape[1], "missing_values": int(train.isna().sum().sum())},
            {"dataset": "test", "rows": len(test), "columns": test.shape[1], "missing_values": int(test.isna().sum().sum())},
        ]
        if original is not None:
            rows.append({"dataset": "original", "rows": len(original), "columns": original.shape[1], "missing_values": int(original.isna().sum().sum())})
        return pd.DataFrame(rows)

    def feature_group_report(self):
        freq_feature_count = sum(
            1
            for col in self.feature_cols
            if col.endswith("_count") or col.endswith("_freq")
        )
        return pd.DataFrame(
            [
                {"group": "all_features", "count": len(self.feature_cols)},
                {"group": "categorical", "count": len(self.cat_cols)},
                {"group": "numeric", "count": len(self.num_cols)},
                {"group": "frequency_source_columns", "count": len(self.frequency_cols)},
                {"group": "count_frequency_features", "count": freq_feature_count},
                {"group": "group_stat_features", "count": len(self.group_stat_features)},
                {"group": "digit_source_columns", "count": len(self.digit_source_cols)},
                {"group": "digit_features", "count": len(self.digit_features)},
                {"group": "signature_features", "count": len(self.signature_features)},
                {"group": "string_precision_features", "count": len(self.string_precision_features)},
            ]
        )


feature_engineer = FeatureEngineer(cfg)
train_fe, test_fe, original_fe = feature_engineer.transform_all(train_raw, test_raw, original_raw)

show_title("Feature builder", "Domain, signature, cross, frequency, and group-stat features")
show_record_cards("Feature datasets", feature_engineer.feature_report(train_fe, test_fe, original_fe), max_rows=3, columns=3)
show_metric_cards("Feature groups", dict(zip(feature_engineer.feature_group_report()["group"], feature_engineer.feature_group_report()["count"])), columns=5)
show_feature_list("Frequency source columns", feature_engineer.frequency_cols, max_items=30)
show_feature_list("Group statistic features", feature_engineer.group_stat_features, max_items=24)
if hasattr(feature_engineer, "digit_source_cols"):
    show_feature_list("Digit source columns", feature_engineer.digit_source_cols, max_items=30)
if hasattr(feature_engineer, "signature_features"):
    show_feature_list("Signature features", feature_engineer.signature_features, max_items=30)
if hasattr(feature_engineer, "string_precision_features"):
    show_feature_list("String precision features", feature_engineer.string_precision_features, max_items=10)
show_feature_list("Categorical features", feature_engineer.cat_cols, max_items=35)
```

## [cell 8 MD]
## Matrix Preparation

## [cell 9 CODE]
```python
"""Prepare model matrices and keep the same feature order for validation and test prediction."""

class MatrixBuilder:
    def __init__(self, cfg: Config, feature_engineer: FeatureEngineer):
        self.cfg = cfg
        self.feature_engineer = feature_engineer

    def build(self, train, test, original=None):
        X_comp = train.drop(columns=[self.cfg.target], errors="ignore")
        y_comp = train[self.cfg.target].astype(int)
        X_test = test.copy()

        if original is not None and self.cfg.target in original.columns:
            X_orig = original.drop(columns=[self.cfg.target], errors="ignore")
            y_orig = original[self.cfg.target].astype(int)
        else:
            X_orig, y_orig = None, None

        common_features = [col for col in X_comp.columns if col in X_test.columns]
        X_comp = X_comp[common_features]
        X_test = X_test[common_features]

        if X_orig is not None:
            for col in common_features:
                if col not in X_orig.columns:
                    X_orig[col] = np.nan
            X_orig = X_orig[common_features]

        cat_cols = [col for col in self.feature_engineer.cat_cols if col in common_features]
        cat_features_idx = [X_comp.columns.get_loc(col) for col in cat_cols]
        return X_comp, y_comp, X_test, X_orig, y_orig, cat_cols, cat_features_idx

    def matrix_report(self, X_comp, y_comp, X_test, X_orig, y_orig, cat_cols):
        rows = [
            {"matrix": "X_comp", "rows": len(X_comp), "columns": X_comp.shape[1], "target_rate": y_comp.mean()},
            {"matrix": "X_test", "rows": len(X_test), "columns": X_test.shape[1], "target_rate": np.nan},
        ]
        if X_orig is not None and y_orig is not None:
            rows.append({"matrix": "X_orig", "rows": len(X_orig), "columns": X_orig.shape[1], "target_rate": y_orig.mean()})
        rows.append({"matrix": "categorical_features", "rows": len(cat_cols), "columns": np.nan, "target_rate": np.nan})
        return pd.DataFrame(rows)


matrix_builder = MatrixBuilder(cfg, feature_engineer)
X_comp, y_comp, X_test, X_orig, y_orig, cat_cols, cat_features_idx = matrix_builder.build(train_fe, test_fe, original_fe)

show_title("Matrix preparation", "Aligned matrices for validation, final training, and prediction")
show_record_cards("Model matrices", matrix_builder.matrix_report(X_comp, y_comp, X_test, X_orig, y_orig, cat_cols), max_rows=4, columns=4)
```

## [cell 10 MD]
## CatBoost Validation

## [cell 11 CODE]
```python
"""Train CatBoost with grouped folds and report only the key metrics in simple tables."""

from sklearn.metrics import f1_score, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from catboost import CatBoostClassifier


class CatBoostPitStopModel:
    def __init__(self, cfg: Config, cat_features_idx: list[int]):
        self.cfg = cfg
        self.cat_features_idx = cat_features_idx
        self.best_iters: list[int] = []
        self.fold_metrics: list[dict] = []
        self.oof_pred = None
        self.validation_y_true = None
        self.validation_pred = None
        self.final_model = None
        self.fold_importances: list[np.ndarray] = []

    def params(self, seed, iterations, validation=True):
        params = {
            "iterations": iterations,
            "learning_rate": 0.018,
            "depth": 8,
            "l2_leaf_reg": 8.5,
            "random_strength": 0.65,
            "bootstrap_type": "Bayesian",
            "bagging_temperature": 0.45,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "auto_class_weights": "Balanced",
            "task_type": "GPU",
            "devices": "0:1",
            "random_seed": seed,
            "allow_writing_files": False,
            "verbose": 300,
        }
        if validation:
            params["early_stopping_rounds"] = 500
        return params

    @staticmethod
    def best_threshold(y_true, pred):
        thresholds = np.linspace(0.05, 0.95, 181)
        scores = [f1_score(y_true, pred >= threshold) for threshold in thresholds]
        best_idx = int(np.argmax(scores))
        return float(thresholds[best_idx]), float(scores[best_idx])

    def validate(self, X_comp, y_comp, X_orig=None, y_orig=None, groups=None):
        if groups is None:
            groups = np.arange(len(X_comp))

        splitter = StratifiedGroupKFold(n_splits=self.cfg.n_folds, shuffle=True, random_state=self.cfg.seed)
        self.oof_pred = np.zeros(len(X_comp), dtype=float)
        self.best_iters = []
        self.fold_metrics = []
        self.fold_importances = []

        for fold, (tr_idx, val_idx) in enumerate(splitter.split(X_comp, y_comp, groups=groups), 1):
            X_tr_comp = X_comp.iloc[tr_idx].reset_index(drop=True)
            y_tr_comp = y_comp.iloc[tr_idx].reset_index(drop=True)
            X_val = X_comp.iloc[val_idx].reset_index(drop=True)
            y_val = y_comp.iloc[val_idx].reset_index(drop=True)

            if X_orig is not None and y_orig is not None:
                X_tr = pd.concat([X_tr_comp, X_orig.reset_index(drop=True)], axis=0, ignore_index=True)
                y_tr = pd.concat([y_tr_comp, y_orig.reset_index(drop=True)], axis=0, ignore_index=True)
            else:
                X_tr = X_tr_comp.copy()
                y_tr = y_tr_comp.copy()

            model = CatBoostClassifier(**self.params(seed=self.cfg.seed + fold, iterations=11000, validation=True))
            model.fit(
                X_tr,
                y_tr,
                eval_set=(X_val, y_val),
                cat_features=self.cat_features_idx,
                use_best_model=True,
            )

            val_pred = np.clip(model.predict_proba(X_val)[:, 1], 1e-7, 1 - 1e-7)
            self.oof_pred[val_idx] = val_pred
            best_thr, best_f1 = self.best_threshold(y_val, val_pred)
            y_hat_05 = val_pred >= 0.5
            y_hat_best = val_pred >= best_thr
            best_iter = model.get_best_iteration()
            self.best_iters.append(best_iter)
            self.fold_importances.append(model.get_feature_importance())
            self.fold_metrics.append(
                {
                    "fold": fold,
                    "train_rows": len(X_tr),
                    "valid_rows": len(X_val),
                    "auc": roc_auc_score(y_val, val_pred),
                    "logloss": log_loss(y_val, val_pred),
                    "f1_0_5": f1_score(y_val, y_hat_05),
                    "best_threshold": best_thr,
                    "f1_best": best_f1,
                    "precision_best": precision_score(y_val, y_hat_best),
                    "recall_best": recall_score(y_val, y_hat_best),
                    "best_iter": best_iter,
                }
            )
            gc.collect()

        return pd.DataFrame(self.fold_metrics)

    def validate_single_split(self, X_comp, y_comp, X_orig=None, y_orig=None):
        X_tr_comp, X_val, y_tr_comp, y_val = train_test_split(
            X_comp,
            y_comp,
            test_size=self.cfg.valid_size,
            random_state=self.cfg.seed,
            stratify=y_comp,
        )

        X_tr_comp = X_tr_comp.reset_index(drop=True)
        y_tr_comp = y_tr_comp.reset_index(drop=True)
        X_val = X_val.reset_index(drop=True)
        y_val = y_val.reset_index(drop=True)

        if X_orig is not None and y_orig is not None:
            X_tr = pd.concat([X_tr_comp, X_orig.reset_index(drop=True)], axis=0, ignore_index=True)
            y_tr = pd.concat([y_tr_comp, y_orig.reset_index(drop=True)], axis=0, ignore_index=True)
        else:
            X_tr = X_tr_comp.copy()
            y_tr = y_tr_comp.copy()

        self.oof_pred = np.full(len(X_comp), np.nan, dtype=float)
        self.best_iters = []
        self.fold_metrics = []
        self.fold_importances = []

        model = CatBoostClassifier(**self.params(seed=self.cfg.seed, iterations=11000, validation=True))
        model.fit(
            X_tr,
            y_tr,
            eval_set=(X_val, y_val),
            cat_features=self.cat_features_idx,
            use_best_model=True,
        )

        val_pred = np.clip(model.predict_proba(X_val)[:, 1], 1e-7, 1 - 1e-7)
        best_thr, best_f1 = self.best_threshold(y_val, val_pred)
        y_hat_05 = val_pred >= 0.5
        y_hat_best = val_pred >= best_thr
        best_iter = model.get_best_iteration()

        self.best_iters.append(best_iter)
        self.fold_importances.append(model.get_feature_importance())
        self.validation_y_true = y_val.reset_index(drop=True)
        self.validation_pred = val_pred
        self.fold_metrics.append(
            {
                "fold": "single_split",
                "train_rows": len(X_tr),
                "valid_rows": len(X_val),
                "auc": roc_auc_score(y_val, val_pred),
                "logloss": log_loss(y_val, val_pred),
                "f1_0_5": f1_score(y_val, y_hat_05),
                "best_threshold": best_thr,
                "f1_best": best_f1,
                "precision_best": precision_score(y_val, y_hat_best),
                "recall_best": recall_score(y_val, y_hat_best),
                "best_iter": best_iter,
            }
        )
        return pd.DataFrame(self.fold_metrics)

    def validation_feature_importance(self, feature_names):
        if not self.fold_importances:
            return pd.DataFrame(columns=["feature", "importance"])
        importance = np.mean(np.vstack(self.fold_importances), axis=0)
        return pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values("importance", ascending=False)

    def overall_report(self, y_true):
        if self.oof_pred is not None and not np.isnan(self.oof_pred).any():
            report_y = y_true
            report_pred = self.oof_pred
        else:
            report_y = self.validation_y_true
            report_pred = self.validation_pred

        best_thr, best_f1 = self.best_threshold(report_y, report_pred)
        y_hat_best = report_pred >= best_thr
        return pd.DataFrame(
            [
                {
                    "auc": roc_auc_score(report_y, report_pred),
                    "logloss": log_loss(report_y, report_pred),
                    "best_threshold": best_thr,
                    "f1_best": best_f1,
                    "precision_best": precision_score(report_y, y_hat_best),
                    "recall_best": recall_score(report_y, y_hat_best),
                    "mean_best_iter": np.mean(self.best_iters),
                }
            ]
        )

    @staticmethod
    def prediction_summary(pred):
        return pd.DataFrame(
            [
                {
                    "min": pred.min(),
                    "p01": np.percentile(pred, 1),
                    "p05": np.percentile(pred, 5),
                    "p25": np.percentile(pred, 25),
                    "median": np.median(pred),
                    "p75": np.percentile(pred, 75),
                    "p95": np.percentile(pred, 95),
                    "p99": np.percentile(pred, 99),
                    "max": pred.max(),
                    "mean": pred.mean(),
                }
            ]
        )

    @staticmethod
    def feature_importance(model, feature_names):
        return pd.DataFrame(
            {
                "feature": feature_names,
                "importance": model.get_feature_importance(),
            }
        ).sort_values("importance", ascending=False)

    def fit_final(self, X_comp, y_comp, X_orig=None, y_orig=None):
        if not self.best_iters:
            raise RuntimeError("Run validation before fitting the final model.")

        final_iterations = max(1800, int(np.mean(self.best_iters) * 1.12))
        if X_orig is not None and y_orig is not None:
            X_full = pd.concat([X_comp.reset_index(drop=True), X_orig.reset_index(drop=True)], axis=0, ignore_index=True)
            y_full = pd.concat([y_comp.reset_index(drop=True), y_orig.reset_index(drop=True)], axis=0, ignore_index=True)
        else:
            X_full = X_comp.copy()
            y_full = y_comp.copy()

        self.final_model = CatBoostClassifier(
            **self.params(seed=self.cfg.seed + 999, iterations=final_iterations, validation=False)
        )
        self.final_model.fit(X_full, y_full, cat_features=self.cat_features_idx)
        final_stats = pd.DataFrame(
            [
                {
                    "train_rows": len(X_full),
                    "test_features": X_full.shape[1],
                    "final_iterations": final_iterations,
                    "target_rate": y_full.mean(),
                }
            ]
        )
        return self.final_model, X_full, final_stats
```

## [cell 12 CODE]
```python
"""Run validation when requested, or reuse fixed final iterations for faster reruns."""

trainer = CatBoostPitStopModel(cfg, cat_features_idx)

if cfg.run_validation:
    if cfg.validation_mode == "single_split":
        fold_metrics = trainer.validate_single_split(X_comp, y_comp, X_orig=X_orig, y_orig=y_orig)
        validation_prediction_summary = trainer.prediction_summary(trainer.validation_pred)
    elif cfg.validation_mode == "group_kfold":
        groups = train_raw["Race"].astype(str) + "_" + train_raw["Year"].astype(str)
        fold_metrics = trainer.validate(X_comp, y_comp, X_orig=X_orig, y_orig=y_orig, groups=groups)
        validation_prediction_summary = trainer.prediction_summary(trainer.oof_pred)
    else:
        raise ValueError(f"Unknown validation_mode: {cfg.validation_mode}")

    overall_metrics = trainer.overall_report(y_comp)
    validation_feature_importance = trainer.validation_feature_importance(X_comp.columns)
    selected_final_iterations = max(1800, int(float(overall_metrics.loc[0, "mean_best_iter"]) * 1.12))

    show_title("Validation", f"Mode: {cfg.validation_mode}")
    show_record_cards("Validation split", fold_metrics.round(6), max_rows=len(fold_metrics), columns=1)
    show_metric_cards("Validation metrics", overall_metrics.round(6), columns=4)
    show_metric_cards("Validation prediction distribution", validation_prediction_summary.round(6), columns=5)
    show_ranked_features("Top validation feature importance", validation_feature_importance, max_rows=25)
else:
    if cfg.fixed_final_iterations is None:
        raise ValueError("Set cfg.fixed_final_iterations or enable cfg.run_validation.")

    selected_final_iterations = int(cfg.fixed_final_iterations)
    trainer.best_iters = [selected_final_iterations / 1.12]
    fold_metrics = pd.DataFrame(
        [
            {
                "validation_mode": cfg.validation_mode,
                "run_validation": False,
                "fixed_final_iterations": selected_final_iterations,
            }
        ]
    )
    overall_metrics = pd.DataFrame(
        [
            {
                "auc": np.nan,
                "logloss": np.nan,
                "best_threshold": np.nan,
                "f1_best": np.nan,
                "precision_best": np.nan,
                "recall_best": np.nan,
                "mean_best_iter": selected_final_iterations / 1.12,
            }
        ]
    )
    validation_prediction_summary = pd.DataFrame()
    validation_feature_importance = pd.DataFrame({"feature": X_comp.columns, "importance": np.nan})

    show_title("Validation skipped", "Using fixed final iterations from the previous validated run")
    show_metric_cards(
        "Fixed training plan",
        {
            "run_validation": cfg.run_validation,
            "final_iterations": selected_final_iterations,
            "validation_mode": cfg.validation_mode,
            "reference_best_iter": selected_final_iterations / 1.12,
        },
        columns=4,
    )
```

## [cell 13 MD]
## Final Seed Ensemble and Files

## [cell 14 CODE]
```python
"""Fit the two selected final CatBoost models, average predictions, and save output files."""

final_iterations = selected_final_iterations

if X_orig is not None and y_orig is not None:
    X_full = pd.concat([X_comp.reset_index(drop=True), X_orig.reset_index(drop=True)], axis=0, ignore_index=True)
    y_full = pd.concat([y_comp.reset_index(drop=True), y_orig.reset_index(drop=True)], axis=0, ignore_index=True)
else:
    X_full = X_comp.copy()
    y_full = y_comp.copy()

model_predictions = []
model_importances = []
model_stats = []

for model_idx, seed in enumerate(cfg.ensemble_seeds, 1):
    params = trainer.params(seed=seed + 999, iterations=final_iterations, validation=False)
    final_model = CatBoostClassifier(**params)
    final_model.fit(X_full, y_full, cat_features=cat_features_idx)

    pred = np.clip(final_model.predict_proba(X_test)[:, 1], 1e-7, 1 - 1e-7)
    model_predictions.append(pred)
    model_importances.append(final_model.get_feature_importance())
    model_stats.append(
        {
            "model": model_idx,
            "seed": seed,
            "iterations": final_iterations,
            "prediction_mean": pred.mean(),
            "prediction_min": pred.min(),
            "prediction_max": pred.max(),
        }
    )

model_prediction_matrix = np.vstack(model_predictions)
test_pred = np.clip(model_prediction_matrix.mean(axis=0), 1e-7, 1 - 1e-7)
model_prediction_std = model_prediction_matrix.std(axis=0)

final_training_stats = pd.DataFrame(
    [
        {
            "train_rows": len(X_full),
            "test_features": X_full.shape[1],
            "final_iterations": final_iterations,
            "target_rate": y_full.mean(),
            "n_models": len(cfg.ensemble_seeds),
        }
    ]
)
ensemble_model_stats = pd.DataFrame(model_stats)
test_prediction_summary = trainer.prediction_summary(test_pred)

final_feature_importance = pd.DataFrame(
    {
        "feature": X_full.columns,
        "importance": np.mean(np.vstack(model_importances), axis=0),
    }
).sort_values("importance", ascending=False)

output_root = Path("outputs")
main_output_dir = output_root / "main"
main_output_dir.mkdir(parents=True, exist_ok=True)

submission = sample_submission.copy()
target_col = cfg.target if cfg.target in submission.columns else [col for col in submission.columns if col != cfg.id_col][0]
submission[target_col] = test_pred
submission.to_csv("submission.csv", index=False)
submission.to_csv(main_output_dir / "sub.csv", index=False)

best_threshold = float(overall_metrics.loc[0, "best_threshold"]) if pd.notna(overall_metrics.loc[0, "best_threshold"]) else np.nan
if cfg.run_validation and trainer.oof_pred is not None and not np.isnan(trainer.oof_pred).any():
    validation_output = pd.DataFrame(
        {
            "y_true": y_comp.values,
            "pred": trainer.oof_pred,
            "class_0_5": (trainer.oof_pred >= 0.5).astype(int),
            "class_best": (trainer.oof_pred >= best_threshold).astype(int),
        }
    )
elif cfg.run_validation:
    validation_output = pd.DataFrame(
        {
            "y_true": trainer.validation_y_true.values,
            "pred": trainer.validation_pred,
            "class_0_5": (trainer.validation_pred >= 0.5).astype(int),
            "class_best": (trainer.validation_pred >= best_threshold).astype(int),
        }
    )
else:
    validation_output = pd.DataFrame(
        [
            {
                "message": "validation skipped",
                "fixed_final_iterations": final_iterations,
            }
        ]
    )
validation_output.to_csv(main_output_dir / "valid_pred.csv", index=False)
validation_feature_importance.to_csv(main_output_dir / "valid_fi.csv", index=False)
final_feature_importance.to_csv(main_output_dir / "final_fi.csv", index=False)
ensemble_model_stats.to_csv(main_output_dir / "models.csv", index=False)

prediction_blend_report = pd.DataFrame(
    [
        {
            "prediction_std_mean": model_prediction_std.mean(),
            "prediction_std_p95": np.percentile(model_prediction_std, 95),
            "prediction_std_max": model_prediction_std.max(),
        }
    ]
)

diagnostics = pd.DataFrame(
    [
        {
            "oof_auc": float(overall_metrics.loc[0, "auc"]),
            "oof_logloss": float(overall_metrics.loc[0, "logloss"]),
            "best_threshold": best_threshold,
            "mean_best_iter": float(overall_metrics.loc[0, "mean_best_iter"]),
            "final_iterations": final_iterations,
            "validation_mode": cfg.validation_mode,
            "run_validation": cfg.run_validation,
            "ensemble_seeds": ",".join(map(str, cfg.ensemble_seeds)),
            "n_models": len(cfg.ensemble_seeds),
            "train_rows": len(X_full),
            "n_features": X_full.shape[1],
            "n_categorical": len(cat_cols),
            "n_frequency_sources": len(feature_engineer.frequency_cols),
            "n_group_stat_features": len(feature_engineer.group_stat_features),
            "n_digit_source_columns": len(feature_engineer.digit_source_cols),
            "n_digit_features": len(feature_engineer.digit_features),
            "n_signature_features": len(feature_engineer.signature_features),
            "n_string_precision_features": len(feature_engineer.string_precision_features),
            "test_pred_mean": float(test_pred.mean()),
            "test_pred_std_mean": float(model_prediction_std.mean()),
        }
    ]
)
diagnostics.to_csv(main_output_dir / "diag.csv", index=False)

saved_files = pd.DataFrame(
    [
        {"file": "submission.csv", "folder": ".", "rows": len(submission)},
        {"file": "sub.csv", "folder": "outputs/main", "rows": len(submission)},
        {"file": "valid_pred.csv", "folder": "outputs/main", "rows": len(validation_output)},
        {"file": "valid_fi.csv", "folder": "outputs/main", "rows": len(validation_feature_importance)},
        {"file": "final_fi.csv", "folder": "outputs/main", "rows": len(final_feature_importance)},
        {"file": "models.csv", "folder": "outputs/main", "rows": len(ensemble_model_stats)},
        {"file": "diag.csv", "folder": "outputs/main", "rows": len(diagnostics)},
    ]
)

show_title("Final seed ensemble", "Selected CatBoost seeds: 42 and 777")
show_metric_cards("Final training", final_training_stats.round(6), columns=5)
show_record_cards("Ensemble models", ensemble_model_stats.round(6), max_rows=len(ensemble_model_stats), columns=3)
show_metric_cards("Test prediction distribution", test_prediction_summary.round(6), columns=5)
show_metric_cards("Blend stability", prediction_blend_report.round(6), columns=3)
show_ranked_features("Top final feature importance", final_feature_importance, max_rows=25)
show_record_cards("Diagnostics", diagnostics.round(6), max_rows=1, columns=1)
show_record_cards("Saved files", saved_files, max_rows=len(saved_files), columns=3)
```

## [cell 15 MD]
## Compact Blend Inputs


## [cell 16 CODE]
```python
"""Save only the compact seed files that are useful for downstream blending."""

if "model_prediction_matrix" not in globals():
    raise RuntimeError("Run the final seed ensemble cell before saving compact blend inputs.")

if model_prediction_matrix.ndim != 2:
    raise ValueError("model_prediction_matrix must have shape: n_models x n_test_rows.")

if model_prediction_matrix.shape[0] != len(cfg.ensemble_seeds):
    raise ValueError("Number of prediction rows does not match cfg.ensemble_seeds.")

blend_output_dir = Path("outputs") / "blends"
blend_output_dir.mkdir(parents=True, exist_ok=True)


def save_submission_variant(filepath, pred, variant=None):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    sub = sample_submission.copy()
    target_col = cfg.target if cfg.target in sub.columns else [col for col in sub.columns if col != cfg.id_col][0]
    sub[target_col] = np.clip(pred, 1e-7, 1 - 1e-7)
    sub.to_csv(filepath, index=False)
    return {
        "variant": variant or filepath.stem,
        "folder": str(filepath.parent),
        "file": filepath.name,
        "mean": sub[target_col].mean(),
        "std": sub[target_col].std(),
        "min": sub[target_col].min(),
        "max": sub[target_col].max(),
    }

prob_mean_pred = np.clip(model_prediction_matrix.mean(axis=0), 1e-7, 1 - 1e-7)
compact_predictions = {"prob.csv": prob_mean_pred}
for seed, pred in zip(cfg.ensemble_seeds, model_prediction_matrix):
    compact_predictions[f"s{seed}.csv"] = np.clip(pred, 1e-7, 1 - 1e-7)

compact_rows = []
for filename, pred in compact_predictions.items():
    compact_rows.append(save_submission_variant(blend_output_dir / filename, pred))

# Keep the main Kaggle file on the validated two-seed probability mean.
save_submission_variant("submission.csv", compact_predictions["prob.csv"], variant="kaggle_main")

compact_summary = pd.DataFrame(compact_rows).sort_values(["folder", "file"]).reset_index(drop=True)
compact_summary.to_csv(blend_output_dir / "summary.csv", index=False)

show_title("Compact blend inputs", "Only the useful two-seed probability mean and individual seed files are saved")
show_record_cards("Saved compact files", compact_summary.round(6), max_rows=len(compact_summary), columns=3)
show_metric_cards(
    "Main submission",
    {
        "source_variant": "prob.csv",
        "rows": len(sample_submission),
        "mean": compact_predictions["prob.csv"].mean(),
        "std": compact_predictions["prob.csv"].std(),
    },
    columns=4,
)
```

## [cell 17 MD]
## Meta CatBoost Stacker

The last set of blend experiments gave a clear result: changing the `0.95381` weight around `5%` and adding tiny direct CatBoost variant weights did not beat the current best blend. That means another similar seed or a lightly changed parameter set is unlikely to add enough new information. The next useful experiment is to make our CatBoost signal more structured before it reaches the blender.

This block trains three base CatBoost views and uses their out-of-fold predictions to train a small second-level CatBoost model. The views are deliberately different: `base` uses all engineered features, `clean` removes digit/signature/string precision features, and `domain` also removes group-stat features. The meta-model then learns how to combine these views using OOF predictions, prediction means, disagreement features, and rank features.

The expected output is not a large pack of submissions. The key file is `outputs/stacking/meta.csv`. If the stacker learns a cleaner ranking than the raw two-seed ensemble, `meta.csv` can become a better tiny-weight CatBoost signal for the final blender than the previous `shallow` and `reg` files.


## [cell 18 CODE]
```python
"""Build OOF base predictions and train a small CatBoost meta-model on top of them."""

from sklearn.model_selection import StratifiedKFold

stacking_output_dir = Path("outputs") / "stacking"
stacking_output_dir.mkdir(parents=True, exist_ok=True)


def available_columns(columns):
    return [col for col in columns if col in X_comp.columns]


def feature_view_columns(view_name):
    digit_cols = available_columns(getattr(feature_engineer, "digit_features", []))
    signature_cols = available_columns(getattr(feature_engineer, "signature_features", []))
    string_cols = available_columns(getattr(feature_engineer, "string_precision_features", []))
    group_stat_cols = available_columns(getattr(feature_engineer, "group_stat_features", []))

    drop_cols = set()
    if view_name in {"clean", "domain"}:
        drop_cols.update(digit_cols)
        drop_cols.update(signature_cols)
        drop_cols.update(string_cols)
    if view_name == "domain":
        drop_cols.update(group_stat_cols)

    return [col for col in X_comp.columns if col not in drop_cols]


def subset_cat_indices(columns):
    return [idx for idx, col in enumerate(columns) if col in cat_cols]


def build_full_training_frame(columns):
    if X_orig is not None and y_orig is not None:
        X_full_view = pd.concat(
            [X_comp[columns].reset_index(drop=True), X_orig[columns].reset_index(drop=True)],
            axis=0,
            ignore_index=True,
        )
        y_full_view = pd.concat([y_comp.reset_index(drop=True), y_orig.reset_index(drop=True)], axis=0, ignore_index=True)
    else:
        X_full_view = X_comp[columns].copy()
        y_full_view = y_comp.copy()
    return X_full_view, y_full_view


def catboost_params_for_stack(seed, iterations, validation):
    params = trainer.params(seed=seed, iterations=iterations, validation=validation)
    params["verbose"] = 500
    return params


def train_oof_and_test_for_view(view_name, seed_offset):
    columns = feature_view_columns(view_name)
    view_cat_features = subset_cat_indices(columns)
    splitter = StratifiedKFold(n_splits=cfg.stacking_n_folds, shuffle=True, random_state=cfg.seed + seed_offset)

    oof_pred = np.zeros(len(X_comp), dtype=float)
    fold_rows = []
    best_iters = []

    for fold, (tr_idx, val_idx) in enumerate(splitter.split(X_comp[columns], y_comp), 1):
        X_tr_comp = X_comp.iloc[tr_idx][columns].reset_index(drop=True)
        y_tr_comp = y_comp.iloc[tr_idx].reset_index(drop=True)
        X_val = X_comp.iloc[val_idx][columns].reset_index(drop=True)
        y_val = y_comp.iloc[val_idx].reset_index(drop=True)

        if X_orig is not None and y_orig is not None:
            X_tr = pd.concat([X_tr_comp, X_orig[columns].reset_index(drop=True)], axis=0, ignore_index=True)
            y_tr = pd.concat([y_tr_comp, y_orig.reset_index(drop=True)], axis=0, ignore_index=True)
        else:
            X_tr = X_tr_comp.copy()
            y_tr = y_tr_comp.copy()

        model = CatBoostClassifier(
            **catboost_params_for_stack(
                seed=cfg.seed + seed_offset + fold,
                iterations=cfg.stacking_base_iterations,
                validation=True,
            )
        )
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), cat_features=view_cat_features, use_best_model=True)
        val_pred = np.clip(model.predict_proba(X_val)[:, 1], 1e-7, 1 - 1e-7)
        oof_pred[val_idx] = val_pred
        best_iter = model.get_best_iteration()
        best_iters.append(best_iter)
        fold_rows.append(
            {
                "view": view_name,
                "fold": fold,
                "n_features": len(columns),
                "n_categorical": len(view_cat_features),
                "best_iter": best_iter,
                "auc": roc_auc_score(y_val, val_pred),
                "logloss": log_loss(y_val, val_pred),
            }
        )
        gc.collect()

    final_iterations_view = max(1800, int(np.mean(best_iters) * 1.08)) if best_iters else cfg.stacking_base_iterations
    X_full_view, y_full_view = build_full_training_frame(columns)
    final_model = CatBoostClassifier(
        **catboost_params_for_stack(
            seed=cfg.seed + seed_offset + 999,
            iterations=final_iterations_view,
            validation=False,
        )
    )
    final_model.fit(X_full_view, y_full_view, cat_features=view_cat_features)
    test_pred_view = np.clip(final_model.predict_proba(X_test[columns])[:, 1], 1e-7, 1 - 1e-7)

    pd.DataFrame({cfg.id_col: sample_submission[cfg.id_col].values, view_name: test_pred_view}).to_csv(
        stacking_output_dir / f"test_{view_name}.csv", index=False
    )
    pd.DataFrame({"row_id": np.arange(len(oof_pred)), "y_true": y_comp.values, view_name: oof_pred}).to_csv(
        stacking_output_dir / f"oof_{view_name}.csv", index=False
    )

    return oof_pred, test_pred_view, pd.DataFrame(fold_rows), {
        "view": view_name,
        "n_features": len(columns),
        "n_categorical": len(view_cat_features),
        "mean_best_iter": float(np.mean(best_iters)),
        "final_iterations": final_iterations_view,
        "oof_auc": roc_auc_score(y_comp, oof_pred),
        "oof_logloss": log_loss(y_comp, oof_pred),
        "test_mean": test_pred_view.mean(),
        "test_std": test_pred_view.std(),
    }


base_views = [
    ("base", 1000),
    ("clean", 2000),
    ("domain", 3000),
]

oof_predictions = {}
test_predictions = {}
fold_reports = []
view_rows = []

for view_name, seed_offset in base_views:
    oof_pred, test_pred_view, fold_report, view_row = train_oof_and_test_for_view(view_name, seed_offset)
    oof_predictions[view_name] = oof_pred
    test_predictions[view_name] = test_pred_view
    fold_reports.append(fold_report)
    view_rows.append(view_row)

stack_oof = pd.DataFrame({name: pred for name, pred in oof_predictions.items()})
stack_test = pd.DataFrame({name: pred for name, pred in test_predictions.items()})

for frame in [stack_oof, stack_test]:
    frame["pred_mean"] = frame[["base", "clean", "domain"]].mean(axis=1)
    frame["pred_std"] = frame[["base", "clean", "domain"]].std(axis=1)
    frame["base_clean_diff"] = frame["base"] - frame["clean"]
    frame["base_domain_diff"] = frame["base"] - frame["domain"]
    frame["clean_domain_diff"] = frame["clean"] - frame["domain"]
    for col in ["base", "clean", "domain"]:
        frame[f"{col}_rank"] = pd.Series(frame[col]).rank(method="average", pct=True).to_numpy()

meta_feature_cols = list(stack_oof.columns)
X_meta_train, X_meta_val, y_meta_train, y_meta_val = train_test_split(
    stack_oof[meta_feature_cols],
    y_comp,
    test_size=cfg.valid_size,
    random_state=cfg.seed + 77,
    stratify=y_comp,
)

meta_model = CatBoostClassifier(
    iterations=cfg.stacking_meta_iterations,
    learning_rate=0.025,
    depth=3,
    l2_leaf_reg=20.0,
    random_strength=0.8,
    bootstrap_type="Bayesian",
    bagging_temperature=0.35,
    loss_function="Logloss",
    eval_metric="AUC",
    task_type="GPU",
    devices="0:1",
    random_seed=cfg.seed + 909,
    allow_writing_files=False,
    early_stopping_rounds=300,
    verbose=300,
)
meta_model.fit(X_meta_train, y_meta_train, eval_set=(X_meta_val, y_meta_val), use_best_model=True)

meta_oof_like = np.clip(meta_model.predict_proba(stack_oof[meta_feature_cols])[:, 1], 1e-7, 1 - 1e-7)
meta_test_pred = np.clip(meta_model.predict_proba(stack_test[meta_feature_cols])[:, 1], 1e-7, 1 - 1e-7)

submission_meta = sample_submission.copy()
target_col = cfg.target if cfg.target in submission_meta.columns else [col for col in submission_meta.columns if col != cfg.id_col][0]
submission_meta[target_col] = meta_test_pred
submission_meta.to_csv(stacking_output_dir / "meta.csv", index=False)

pd.concat(fold_reports, ignore_index=True).to_csv(stacking_output_dir / "folds.csv", index=False)
stack_oof.assign(y_true=y_comp.values, meta=meta_oof_like).to_csv(stacking_output_dir / "oof_stack.csv", index=False)
stack_test.assign(meta=meta_test_pred).to_csv(stacking_output_dir / "test_stack.csv", index=False)

view_summary = pd.DataFrame(view_rows)
meta_summary = pd.DataFrame(
    [
        {
            "model": "meta_catboost",
            "n_meta_features": len(meta_feature_cols),
            "best_iter": meta_model.get_best_iteration(),
            "holdout_auc": roc_auc_score(y_meta_val, np.clip(meta_model.predict_proba(X_meta_val)[:, 1], 1e-7, 1 - 1e-7)),
            "holdout_logloss": log_loss(y_meta_val, np.clip(meta_model.predict_proba(X_meta_val)[:, 1], 1e-7, 1 - 1e-7)),
            "full_oof_like_auc": roc_auc_score(y_comp, meta_oof_like),
            "full_oof_like_logloss": log_loss(y_comp, meta_oof_like),
            "test_mean": meta_test_pred.mean(),
            "test_std": meta_test_pred.std(),
            "corr_with_base_test": np.corrcoef(meta_test_pred, test_predictions["base"])[0, 1],
            "mean_abs_delta_vs_base_test": np.abs(meta_test_pred - test_predictions["base"]).mean(),
        }
    ]
)
summary = pd.concat([view_summary, meta_summary], ignore_index=True, sort=False)
summary.to_csv(stacking_output_dir / "summary.csv", index=False)

show_title("Meta CatBoost stacker", "OOF stacking over compact CatBoost feature-family predictions")
show_record_cards("Base view summary", view_summary.round(6), max_rows=len(view_summary), columns=1)
show_record_cards("Meta model summary", meta_summary.round(6), max_rows=1, columns=1)
show_metric_cards(
    "Stacking output",
    {
        "base_views": len(base_views),
        "meta_features": len(meta_feature_cols),
        "folder": str(stacking_output_dir),
        "submission": "meta.csv",
    },
    columns=4,
)
```
