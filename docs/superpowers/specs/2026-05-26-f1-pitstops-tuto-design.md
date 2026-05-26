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
- **Light ISO/SOTA cadre** : not the orbit-wars 14-ISO stack, but the minimum-non-negotiable spelled out in §13 (lifecycle traceability, seed-everything, group-aware CV, paired bench, calibration audit, anti-AI-slop).
- **Session-end discipline from day-1** : skill `project-session-end` projected at bootstrap (§14), T1-T5 triggers active throughout the project (not added after-the-fact).

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
| 00 | setup | **~2h (extended, see §12)** | Bootstrap pro : venv + pyproject + pre-commit + pre-push + context7-validated deps + session-end skill projection + CLAUDE.md projet + memory bootstrap. |
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
# Runtime
python 3.11
pandas, numpy, scikit-learn
catboost, lightgbm
matplotlib, seaborn
torch, pycox            # for step 04 if survival
lifelines               # CoxPH baseline comparison
jupyter, ipywidgets
kaggle                  # CLI

# Dev (PEP 735 dependency-groups)
ruff                    # format + lint
mypy                    # type check (selective, src/ only — not notebooks)
pre-commit              # hooks orchestrator
pytest                  # sanity asserts harness
nbstripout              # clean notebook outputs from git
detect-secrets          # security baseline
```

**Versions pinned via context7 MCP query before install** (per CLAUDE.md root "Toujours rechercher avant d'affirmer" — anti-stale-version-from-training-data). Concrete versions land in `pyproject.toml` + `requirements.lock` at step 00 execution time, not pre-decided here.

Deliberately excluded: MLflow, DVC, Optuna (heavy), Weights & Biases. Out of scope for a 6-notebook tutorial.

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

- Repo skeleton scaffolded with venv + deps installed + **all hooks active** (pre-commit + pre-push).
- **Session-end skill projected day-1** (§14) + first invocation tested.
- **ISO/SOTA ML standards (§13)** referenced and applied from notebook 02 onwards.
- Notebooks 00-04 executed end-to-end, each producing one submission + one tracking row.
- README tracking table populated with 3+ rows (baseline + FE + at least one of {temporal, MoE}).
- Each step committed with `submit-NN` tag and message containing CV/LB scores.
- Brief retrospective added at the end of README : "what I learned per step + what surprised me".
- Step 05 done if time permits ; not a hard requirement.

---

## 12. Setup dev environment (step 00 extended — was undersized)

The original "10 min" estimate was naive. A proper bootstrap is ~2h and includes the items below. **All of step 12 lives inside notebook 00 + accompanying shell scripts, NOT spread across the parcours.**

### 12.1 Repo & version control

- `git init` + initial commit of skeleton (already done : commit `a5f12e2` for design spec)
- `.gitignore` : `.venv/`, `data/raw/`, `data/processed/`, `__pycache__/`, `.ipynb_checkpoints/`, `submissions/*.csv`, `.DS_Store`, `*.pyc`, `.mypy_cache/`, `.ruff_cache/`
- `.gitattributes` : declare `*.ipynb` as `linguist-detectable=true` + `text eol=lf`

### 12.2 Python venv + deps (pinned via context7)

- Python 3.11 venv : `python -m venv .venv` ; activation Windows : `.venv\Scripts\activate`
- **Use context7 MCP** to validate latest stable versions of each lib BEFORE pinning (anti-stale-version, per CLAUDE.md root "Toujours rechercher avant d'affirmer") :
  - `pandas`, `numpy`, `scikit-learn`, `catboost`, `lightgbm`, `matplotlib`, `seaborn`
  - `torch` (CPU wheel ok, no CUDA for tuto), `pycox`, `lifelines`
  - `jupyter`, `ipywidgets`, `nbstripout`
  - `kaggle` CLI
  - Dev : `ruff`, `mypy`, `pre-commit`, `pytest`
- `pyproject.toml` PEP 621 format with `[project]` + `[tool.ruff]` + `[tool.mypy]` + `[dependency-groups]` (PEP 735 dev deps)
- `requirements.lock` generated via `pip freeze > requirements.lock` after install (reproducibility)

### 12.3 Pre-commit hooks (`.pre-commit-config.yaml`)

Minimum viable set (no overkill) :

- `ruff format` (formatting)
- `ruff check --fix` (linting)
- `nbstripout` (clean notebook outputs from git — re-runnable)
- `detect-secrets` or `gitleaks` (security baseline)
- `check-yaml`, `end-of-file-fixer`, `trailing-whitespace` (standard hygiene)
- `check-added-large-files` with `--maxkb=500` (block accidental CSV commits)

### 12.4 Pre-push hooks (lefthook or pre-commit `--hook-type pre-push`)

- Run sanity asserts (notebook 00 dataset assertions executed headless via `jupyter nbconvert --to script` + `python -c`)
- Block push if `requirements.lock` is stale vs `pyproject.toml`
- Block push if `submissions/*.csv` somehow committed (defense in depth vs gitignore)

### 12.5 CLAUDE.md projet

Minimal CLAUDE.md at repo root (`C:/Dev/f1-pitstops-s6e5/CLAUDE.md`) declaring :

- Project intent (tuto-fun, link to design spec)
- STANDING PROCEDURE — POST-TASK MEMORY + FATIGUE SELF-ASSESSMENT (T1-T5 triggers, light version of orbit-wars STANDING PROCEDURE — see §14 for projection)
- Hard signals : isolation from orbit-wars, no LB grinding, time-box discipline, ISO standards level (§13)
- Pointer to memory dir : `C:/Users/pierr/.claude/projects/C--Dev-f1-pitstops-s6e5/memory/`

### 12.6 Memory dir bootstrap

Create `C:/Users/pierr/.claude/projects/C--Dev-f1-pitstops-s6e5/memory/` with :

- `MEMORY.md` (index, empty initially)
- `feedback_session_end_template.md` (= projection of `C:/Dev/feedback_session_end_template_GENERIC.md`, drop `_GENERIC`)
- `feedback_post_task_procedure_and_fatigue.md` (= 6-indicator fatigue table, copied from canonical)
- `feedback_per_task_memory_updates.md` (= per-task memo discipline)
- `feedback_diagnostic_first_doctrine.md` (= anti-pattern counter source-of-truth)
- `project_session_history_archive.md` (created empty, prepended on each session-end)

---

## 13. ISO/SOTA ML standards (light cadre — tuto-aware)

**Not** the full 14-ISO orbit-wars stack (overkill for 6 notebooks). But enough cadre that the user **learns the discipline** and the code doesn't drift into AI-slop. References below are aspirational — we read the abstract, we adopt the spirit, we don't write a conformity matrix.

### 13.1 Lifecycle traceability — ISO/IEC 23053:2022 (ML framework)

- Each notebook = one numbered lifecycle phase (`00_setup`, `01_eda`, `02_baseline`, ...).
- Each notebook ends with a "what was decided" block that becomes the next notebook's input.
- Submission CSVs named with phase prefix : `02_baseline_catboost.csv`, `03_domain_fe.csv`, ...
- Git commits tagged `submit-NN-<approach>` mapping commit hash ↔ ladder identity.

### 13.2 Software quality — ISO/IEC 25010 (light subset)

| Sub-characteristic | How we apply it (tuto-level) |
|---|---|
| **Functional suitability** | Notebooks runnable end-to-end on a fresh kernel restart. Sanity asserts at top of each. |
| **Reliability — fault tolerance** | `try/except` only at I/O boundaries (kaggle download, file read). NEVER inside ML logic. |
| **Performance** | Each notebook tracks wall-time per major cell. Pyplot off, `engine='c'` on read_csv, batch operations. |
| **Maintainability** | Functions live in `src/`, NOT in notebooks. Notebooks call `src.data.load_train()`, never re-implement. |
| **Portability** | venv + `requirements.lock` + Windows-friendly paths only. No `/tmp`, no `~` in code. |

### 13.3 SOTA ML rigor (the minimum-non-negotiable)

- **Seed everything** : `np.random.seed(42)`, `torch.manual_seed(42)`, `catboost(random_seed=42)`, `random.seed(42)`. Reproducible runs.
- **Group-aware CV** : `GroupKFold(n_splits=5, groups=train['Race']+'-'+train['Year'].astype(str))`. Documented in `src/data.py`. **Never naive KFold.**
- **Out-of-fold (OOF) predictions** stored and used for calibration plots + stacking.
- **Paired bench** : every new approach is compared CV-fold-by-CV-fold against the baseline (same folds), report mean ± std + McNemar / paired t-test p-value if delta < 1pp.
- **Sample size note** : if N (positives in fold) < 1000, flag it in the notebook narrative — gains below ~2pp not significant.
- **Calibration audit** : every model produces `evaluate.plot_calibration()` (reliability diagram + Brier score). For pit-stop prediction calibration matters more than rank order.
- **Feature importance reproducible** : SHAP or permutation, NOT model.feature_importances_ alone (biased toward high-cardinality).

### 13.4 Anti-AI-slop checklist (per CLAUDE.md root)

- No "TBD", no "expected", no "should be X" in notebooks or commit bodies — only measured values.
- No "future-proofing" code (no feature flags, no abstract base classes "in case we add more models").
- No copy-paste constants from orbit-wars without a paired-bench in THIS context first.
- Every claim ("FE helped", "temporal worked") backed by a number in the tracking table.

### 13.5 Out-of-scope ISO normes (explicitly NOT pursued)

- ISO/IEC 42001 (AI management system) — overkill, no governance layer needed for 5-notebook tuto.
- ISO/IEC 5259 / 5338 / 24027 / 29119 / 5469 — orbit-wars-specific, out of scope here.

---

## 14. Session-end skill projection (greenfield profile, day-1)

Per `C:/Dev/skills-templates/project-session-end/README.md` workflow.

### 14.1 Profile classification

**Greenfield** : no existing CLAUDE.md, no existing feedback memory, no existing memo discipline. → Full projection workflow (Étapes 1-5 standard).

### 14.2 Files to project (Étape 0 audit confirms NO existing overlap)

| Source canonique | Projected path | Action |
|---|---|---|
| `C:/Dev/feedback_session_end_template_GENERIC.md` | `<memory-dir>/feedback_session_end_template.md` | Copy + drop `_GENERIC` suffix |
| `C:/Dev/skills-templates/project-session-end/SKILL.md` | `C:/Dev/f1-pitstops-s6e5/.claude/skills/project-session-end/SKILL.md` | Copy + drop `-GENERIC` from frontmatter `name` |
| 3 slow-changing cross-refs (fatigue + per-task + diagnostic-first) | `<memory-dir>/` | Copy from canonical |

### 14.3 19 markers to customize

Per README ch. "Workflow projection chameleon — Étape 2", customize :

- `<PROJECT-SPECIFIC: project-name>` → `f1-pitstops-s6e5`
- `<PROJECT-SPECIFIC: stack-hint>` → `"Stack-aware : Python 3.11 + .venv dédié + Kaggle Playground tabular CSV submission, bench gates = paired CV-fold + LB delta vs baseline."`
- `<PROJECT-SPECIFIC: project-claude-md-path>` → `C:/Dev/f1-pitstops-s6e5/CLAUDE.md`
- `<PROJECT-SPECIFIC: memory-dir>` → `C:/Users/pierr/.claude/projects/C--Dev-f1-pitstops-s6e5/memory/`
- `<PROJECT-SPECIFIC: working-dir-cd>` → `cd C:/Dev/f1-pitstops-s6e5`
- `<PROJECT-SPECIFIC: build-check-cmd>` → `N/A — Python pure, no build step`
- `<PROJECT-SPECIFIC: test-runner-cmd>` → `.venv/Scripts/python -m pytest -q --no-header tests/ 2>&1 | tail -3` (tests minimal — sanity asserts only)
- `<PROJECT-SPECIFIC: lint-cmd>` → `.venv/Scripts/python -m ruff check . 2>&1 | tail -3`
- `<PROJECT-SPECIFIC: extra-state-cmd>` → `kaggle competitions submissions playground-series-s6e5 -v | head -5`
- `<PROJECT-SPECIFIC: task-tag-examples>` → `notebook-02-baseline-shipped`, `notebook-03-domain-fe-cv-0.812`, `submit-05-moe-bonus`
- `<PROJECT-SPECIFIC: T4-milestone-condition>` → `"notebook NN complété end-to-end OU submission Kaggle shippée OU décision archi (e.g. survival vs sequence en notebook 04)"`
- `<PROJECT-SPECIFIC: hard-signals-examples>` → `"isolation from orbit-wars ; no LB grinding ; time-box 1.5x max per step ; seed fix everywhere ; group-aware CV mandatory ; calibration plot per submit"`
- `<PROJECT-SPECIFIC: §6-row6-quality-tools>` → `"pytest + ruff + pre-commit (auto on commit)"`
- `<PROJECT-SPECIFIC: §6-row7-runtime-state>` → `"Kaggle submissions : sub_id + score + tags + COMPLETE/ERROR/PENDING ; local : last notebook executed + CV scores per fold"`
- `<PROJECT-SPECIFIC: §6-row8-mapping-target>` → `"submission_id"`
- `<PROJECT-SPECIFIC: §6-row9-contract>` → `"submit si CV improves baseline by >= 0.5pp ET paired fold-by-fold majority positive ET LB delta cohérent dans 1pp du CV delta"`
- `<PROJECT-SPECIFIC: §6-row10-plans-source>` → `"docs/superpowers/specs/2026-05-26-f1-pitstops-tuto-design.md + future docs/superpowers/plans/"`
- `<PROJECT-SPECIFIC: §6-row13-layer-phase>` → `"Notebook 00-05 status (DONE/IN-PROGRESS/PENDING)"`
- `<PROJECT-SPECIFIC: project-specific-cross-refs>` → pointers to wiki F1 articles (if any) + Kaggle competition page + tracking table in README.md

### 14.4 CLAUDE.md wiring (§Étape 4 of projection workflow)

CLAUDE.md projet (§12.5) declares the STANDING PROCEDURE block pointing to the projected skill — light version, since the project is small. T1-T5 triggers active.

### 14.5 First invocation test

After projection complete, manually trigger the skill once (T5 user-explicit "session-end test") to verify the 7 fresh-Claude-resume self-contained criteria PASS on a real (even if empty) state. **Definition of done §11** depends on this test passing.
