# f1-pitstops-s6e5

Kaggle Playground Series S6E5 — **Predicting F1 Pit Stops** (binary AUC).
Sponsor : Google LLC. Tier : Swag (1 t-shirt). Deadline : **2026-05-31 23:59 UTC**.

## Goal

Submit a CSV of probabilistic predictions (`id,PitNextLap ∈ [0,1]`) for the 188 165 test rows. AUC scoring. 10 submissions/day allowed, 2 final selectable for private leaderboard.

Not a leaderboard hunt — pedagogical journey through tabular ML techniques on a real synthetic dataset, with the journey documented in commits + this README.

## Stack

| Layer | Tool |
|---|---|
| Python | 3.13.13 venv-scoped (`.venv/`) |
| Tabular | pandas 2.3.3, numpy, scikit-learn, scipy |
| Boosting | catboost 1.2.10, lightgbm 4.6.0 |
| Plots | matplotlib, seaborn |
| Kaggle | `kaggle` CLI 2.1.2 (submission via `.venv/Scripts/kaggle.exe`) |
| Dev | ruff, mypy, pre-commit, pytest, detect-secrets |

Pas de jupyter, pas de notebooks, pas de torch/lifelines/pycox. Tout en `.py` purs.

## Setup (one-shot)

```bash
cd C:/Dev/f1-pitstops-s6e5
py -3.13 -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"
.venv/Scripts/python.exe -m pre_commit install
.venv/Scripts/python.exe -m pre_commit install --hook-type pre-push
.venv/Scripts/kaggle.exe competitions download -c playground-series-s6e5 -p data/raw/
# unzip data/raw/playground-series-s6e5.zip in place, then remove the zip
```

## Plan v2 (3 submissions max, J-4)

| # | Approach | Script | What it teaches |
|---|---|---|---|
| 01 | **CatBoost defaults + GroupKFold(Race+Year)** | `scripts/run_01_baseline.py` | Baseline-first principle, group-aware CV, AUC anchor |
| 02 | **+ Row-level FE** (Compound×Stint/Year, TyreLife/Stint, LapTime vs Race-median, target-enc Driver) | `scripts/run_02_row_fe.py` | FE vs hyperparam tuning ROI on tabular |
| 03 | **+ Ensemble LightGBM** or **external F1 dataset augmentation** (time-permitting) | `scripts/run_03_*.py` | Model diversity vs more data |

> **Data constraint** (found 2026-05-28, see `docs/eda/2026-05-28-data-integrity.md`) : the competition dataset is row-shuffled at the `(Race, Year, Driver)` level — **62% of train groups have non-monotonic `Stint`**, 12.75% of consecutive-lap transitions show Stint decreasing. Intra-stint `TyreLife` is preserved (100%). All time-series FE (`laps_since_pit`, rolling lap-time, undercut, stint progression) is **broken** and excluded from submit-02. Only row-level FE.

Ship a submission only if `AUC_CV ≥ baseline + 0.5pp` AND `paired fold majority positive`. CV-LB gap = data point, not a target. Expected fold variance ~±0.010 under GroupKFold (data noise, not bug — confirmed by other competitors).

## Tracking table

| # | Date | Approach | OOF AUC | Public LB | Δ vs baseline | Key insight |
|---|------|----------|---------|-----------|---------------|-------------|
| 01 | 2026-05-28 | CatBoost defaults + GroupKFold(Race+Year) | 0.9294 | **0.94457** | — (anchor) | CV-LB gap +1.52pp (GroupKFold pessimiste); LB rank #1849/2703 (top 68%); median 0.948, top 0.9549 → baseline insuffisant, need FE |
| 02 | TBD | + (Y,R,D) train-lookup aggregates + Aadi Gupta + StratifiedKFold + CatBoost tunés | TBD | TBD | TBD | TBD |
| 03 | TBD | + RealMLP/TabPFN diversity ensemble + LR stacking + isotonic calibration (if time) | TBD | TBD | TBD | TBD |

Filled after each submit. Empty cells = not yet measured (never "expected").

## Run a submission

```bash
# 1. Train + predict + write CSV
.venv/Scripts/python.exe scripts/run_NN_<approach>.py
# (prints OOF AUC per fold + mean ± std + Brier ; writes submissions/NN_<approach>.csv)

# 2. Submit
.venv/Scripts/kaggle.exe competitions submit playground-series-s6e5 \
    -f submissions/NN_<approach>.csv \
    -m "submit-NN: <approach> CV=X.XXX"

# 3. Wait for scoring + fill tracking row
.venv/Scripts/kaggle.exe competitions submissions playground-series-s6e5 | head -5
```

## Repo layout

```
.
├── README.md                    # this file (plan vivant + tracking)
├── CLAUDE.md                    # project rules (Claude only)
├── pyproject.toml               # deps exact-pinned, Python 3.13
├── requirements.lock            # frozen resolved deps (144 pkgs)
├── .pre-commit-config.yaml      # ruff + detect-secrets + lock-sync (pre-push) + pytest (pre-push)
├── data/
│   └── raw/                     # train/test/sample_submission CSVs (gitignored)
├── src/
│   ├── data.py                  # load_train, load_test, make_group_kfold(Race+Year)
│   ├── evaluate.py              # compute_metrics (AUC/logloss/Brier), plot_calibration
│   └── submit.py                # write_submission, submit_to_kaggle
├── tests/                       # 5 tests (3 data + 2 evaluate)
├── scripts/
│   ├── check_lock_sync.py       # pre-push hook
│   └── run_NN_*.py              # one script per submission (created at run time)
├── submissions/                 # NN_<approach>.csv (gitignored)
└── docs/
    ├── research/2026-05-26-deps-versions.md   # deps decisions
    └── superpowers/             # original spec + plan (SUPERSEDED, archive only)
```

## Honesty / retrospective

This project was over-scoped initially (6 notebooks, ISO/SOTA cadre, MoE bonus, survival analysis, session-end skill projection — for a 1-t-shirt prize). Phase 0 bootstrap (Tasks 1-13 of the original plan) was useful infrastructure. Re-scope on 2026-05-27 dropped notebooks + heavy ML deps. The journey from "tuto-fun-as-architecture-fest" to "ship a CSV in 4 days" is itself the lesson.

Tracking is honest : if a submission doesn't beat baseline, the row says so. If we run out of time before submit-03, the row stays at "—".

## License

CC BY 4.0 (matches Kaggle competition data license).
