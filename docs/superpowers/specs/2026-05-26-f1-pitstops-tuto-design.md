# Design — Kaggle Playground S6E5 "Predicting F1 Pit Stops" (tuto-fun mode)

- **Date** : 2026-05-26
- **Author** : pierre-alexandre + Claude
- **Status** : Draft → awaiting user review before plan
- **Companion project** : runs in parallel with `C:/Dev/orbit-wars` ($50k Kaggle competition, deadline 2026-06-23). **Strict isolation** : no shared venv, no cross-imports, no shared time-budget pressure.

---

## 1. Goal

Use Kaggle Playground Series **S6E5 — Predicting F1 Pit Stops** (deadline **2026-05-31 23:59 UTC**, reward: Swag) as a **structured pedagogical playground** for tabular ML.

The objective is **not** a leaderboard ranking. It is to walk a progression of techniques on a real (synthetic-but-realistic) tabular problem and **document what each step teaches**. Output = a working repo + 6 notebooks (1 setup + 5 narrative steps) + a tracking table that captures CV/LB scores and the key insight per step.

If a top-X% ranking happens, fine. If not, fine — the artifact is the journey, not the score.

---

## 2. Dataset (verified 2026-05-26)

- **Task** : binary classification, target `PitNextLap ∈ {0, 1}`
- **Class balance** : ~20% positives (imbalanced but not extreme)
- **Train** : 439 140 rows × 16 columns
- **Test** : 188 165 rows × 15 columns (target removed)
- **Sample submission** : `id, PitNextLap`
- **Features** :
  - Categorical : `Driver` (887 unique, mostly synthetic D###, a few real codes ZON/BEL/YAM), `Compound` (5: HARD/MEDIUM/SOFT/INTERMEDIATE/WET), `Race` (26 real GPs), `Year` (2022-2025)
  - Numerical : `PitStop`, `LapNumber`, `Stint`, `TyreLife`, `Position`, `LapTime (s)`, `LapTime_Delta`, `Cumulative_Degradation`, `RaceProgress`, `Position_Change`
- **Signals observed (EDA preview)** :
  - Pit rate varies **13×** between compounds : WET 2.5% → HARD 32.7% (suggests compound-conditioned modeling has merit)
  - Pit rate vs `RaceProgress` follows an inverted-U, peak ~50% — strong strategic-window pattern
- **Evaluation metric** : not exposed by Kaggle API ; to be confirmed on first submit (hypothesis: AUC or LogLoss, standard Playground binary).

---

## 3. Constraints

- **Isolation from orbit-wars** : dedicated repo `C:/Dev/f1-pitstops-s6e5/`, dedicated venv, dedicated commit history. No shared modules.
- **Time-box** : flexible but capped per step. Step overrun > 1.5× estimate → commit current state + move on.
- **Submission quota** : Kaggle Playground = 5 submissions/day. Plan uses ~1 per step, not grinding.
- **Deadline** : 2026-05-31 23:59 UTC (5 days). If at J-2 we've only shipped baseline + FE, we accept and archive.
- **No production-ML scaffolding** : no MLflow, no DVC, no Optuna-heavy tuning, no unit tests beyond sanity asserts. This is a tutorial repo, not a production codebase.

---

## 4. Repo layout

```
f1-pitstops-s6e5/
├── README.md                       # journal + tracking table (single source of truth)
├── .gitignore                      # data/raw/, .venv/, .ipynb_checkpoints, submissions/*.csv
├── pyproject.toml                  # deps (Python 3.11, pinned)
├── data/
│   ├── raw/                        # train.csv, test.csv, sample_submission.csv (kaggle DL, gitignored)
│   └── processed/                  # FE caches (gitignored)
├── notebooks/
│   ├── 00_setup.ipynb              # bootstrap + download + sanity
│   ├── 01_eda.ipynb                # narrative exploration + visus
│   ├── 02_baseline.ipynb           # CatBoost default + group-aware CV
│   ├── 03_domain_fe.ipynb          # F1-native features
│   ├── 04_temporal.ipynb           # survival OR sequence reframing
│   └── 05_moe_compound.ipynb       # bonus: mixture-of-experts per compound
├── src/
│   ├── __init__.py
│   ├── data.py                     # load_train / load_test / GroupKFold(by=Race+Year)
│   ├── evaluate.py                 # AUC + LogLoss helpers + plot_calibration
│   └── submit.py                   # write CSV → kaggle CLI submit → append README row
├── submissions/
│   └── <NN>_<approach>.csv         # versioned outputs (gitignored, log-only)
└── docs/
    └── superpowers/
        └── specs/
            └── 2026-05-26-f1-pitstops-tuto-design.md   # this file
```

---

## 5. The 5 steps (pedagogy first, score second)

| # | Notebook | Time-box | Pedagogical target |
|---|---|---|---|
| 00 | setup | 10 min | Bootstrap clean : venv, deps, `kaggle competitions download`, sanity asserts. |
| 01 | EDA | 45 min | What questions to ask of a tabular dataset. Domain reading of F1 strategy (compound stints, pit windows, race progress). Visuals: pit rate by compound, by lap, by year, by race. |
| 02 | baseline | 30 min | **Baseline-first principle.** CatBoost defaults on raw features. **Group-aware CV** (`GroupKFold(groups=Race+Year)`) — explain why naive KFold leaks. First submit → anchor LB score + confirm metric. |
| 03 | domain FE | 1 evening | Encoding domain knowledge as features : pit-window-by-compound (Soft 15-20, Med 25-35, Hard 35-45), undercut threat (gap to driver ahead), Year×Compound regulation shifts (2022 aero reset, Pirelli compound revisions), stint progression deltas. Lesson: **FE vs hyperparameter tuning ROI**. |
| 04 | temporal | 1-2 evenings | Reframe tabular as temporal. **Decision tree**: survival analysis (Cox PH via `lifelines`, then DeepSurv via `pycox`) OR sequence modeling (pad-by-stint → LSTM in PyTorch). Choice documented inline based on what step 01 EDA reveals. Lesson: when does a problem stop being "flat tabular". |
| 05 | MoE bonus | 1 evening (optional) | Mixture-of-experts: 5 CatBoost specialists (one per Compound) + logistic meta-stacker. Lesson: conditional architectures vs unified models ; calibration concerns ; stacking pitfalls. |

Each notebook produces:
- 1 submission CSV in `submissions/`
- 1 row appended to the README tracking table
- 1 git commit `submit-NN: <approach> CV=<x> LB=<y>`

---

## 6. Tracking table (lives in README.md)

```markdown
| # | Date | Approach | Features | CV (metric) | LB Public | Δ vs baseline | Key insight |
|---|------|----------|----------|-------------|-----------|---------------|-------------|
| 02 | YYYY-MM-DD | CatBoost default | raw | X.XXX | X.XXX | — | baseline anchored |
| 03 | YYYY-MM-DD | + domain FE | + pit_window, undercut, ... | X.XXX | X.XXX | +Δ | did FE beat tuning? |
| 04 | YYYY-MM-DD | DeepSurv / LSTM | temporal reframe | X.XXX | X.XXX | +Δ | did temporal help? |
| 05 | YYYY-MM-DD | MoE per compound | 5 experts + meta | X.XXX | X.XXX | +Δ | split vs unified |
```

**CV-LB gap is a data point**, not a problem to chase. We note it and learn from it.

---

## 7. Stack (pinned in pyproject.toml at impl time)

```
python 3.11
pandas, numpy, scikit-learn
catboost, lightgbm
matplotlib, seaborn
torch, pycox            # for step 04 if survival
lifelines               # CoxPH baseline comparison
jupyter, ipywidgets
kaggle                  # CLI
```

Deliberately excluded: MLflow, DVC, Optuna (heavy), Weights & Biases. Out of scope for a 5-notebook tutorial.

---

## 8. Scope guards

1. **Time-box discipline**: a step >1.5× its estimate → commit "incomplete but logged" + move on. The journey > any single step.
2. **No orbit-wars contamination**: zero shared imports, zero shared venv, no shared time pressure. If orbit-wars needs attention, this project pauses without ceremony.
3. **CV-LB gap = data point, not a target to hunt**: we record and learn, we do not "tune for the LB".
4. **No submission grinding**: ~1 submit per step, never more. We are not optimizing for ranking.
5. **Deadline behavior**: at J-2 (2026-05-29), whatever is shipped is shipped. No panic sprint. No rushed last-step.

---

## 9. Open questions to resolve during implementation

These are explicitly **not** blockers now ; they will be resolved inline:

1. **Exact evaluation metric** : confirmed at first submit (Kaggle returns score → infer AUC vs LogLoss vs F1).
2. **CV strategy validation** : `GroupKFold(groups=Race+Year)` proposed in step 02 ; we will check fold-stability and revise if variance is too high (might need `GroupKFold(groups=Race+Year+Driver)` to fully prevent leak).
3. **Step 04 framing** : survival (Cox/DeepSurv) **vs** sequence (LSTM) **vs** both. Decided inside notebook 04 based on what EDA in 01 reveals. Documented at decision point.
4. **Step 05 inclusion** : optional. Goes ahead only if step 04 finishes before J-2.

---

## 10. Non-goals

- Top-X% ranking on the leaderboard.
- A reusable production ML pipeline.
- Hyperparameter optimization beyond hand-tuning.
- Ensembling beyond the explicit step 05 MoE.
- Publishing a Kaggle public notebook (kernel push). Submission is CSV-via-API only.

---

## 11. Definition of done

- Repo skeleton scaffolded with venv + deps installed.
- Notebooks 00-04 executed end-to-end, each producing one submission + one tracking row.
- README tracking table populated with 3+ rows (baseline + FE + at least one of {temporal, MoE}).
- Each step committed with `submit-NN` tag and message containing CV/LB scores.
- Brief retrospective added at the end of README : "what I learned per step + what surprised me".
- Step 05 done if time permits ; not a hard requirement.
