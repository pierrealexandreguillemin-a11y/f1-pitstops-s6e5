# CLAUDE.md — `f1-pitstops-s6e5`

> **Scope** : `cwd = C:/Dev/f1-pitstops-s6e5`. Charge en parallèle de `C:/Dev/CLAUDE.md` racine et de `C:/Users/pierr/.claude/projects/C--Dev-f1-pitstops-s6e5/memory/`.

## Intent

Kaggle Playground Series S6E5 — Predicting F1 Pit Stops.

- **Sponsor** : Google LLC
- **Task** : binary classification, prédire probabilité `PitNextLap ∈ [0,1]`
- **Metric** : AUC (ROC)
- **Deadline** : 2026-05-31 23:59 UTC
- **Submissions** : 10/jour, 2 finales sélectionnables, format CSV `id,PitNextLap` (probabilités)
- **Reward** : Swag (T-shirt) — 1 fois par personne dans la série S6
- **Form** : `.py` scripts purs (pas de notebooks, pas de jupyter)

Pédagogie portée par le README narratif + commits explicites, pas par notebooks.

## Stack

- Python 3.13.13 venv-scoped (`.venv/`)
- pandas, numpy, scikit-learn, scipy
- catboost, lightgbm (boosting)
- matplotlib, seaborn (visus README)
- kaggle CLI (submissions)
- dev : ruff, mypy, pre-commit, pytest, detect-secrets

Tout via `.venv/Scripts/python.exe -m <module>` ou `.venv/Scripts/<exe>.exe` direct. Pas de `python` nu, pas de PATH dispatch.

## Hard signals

- **Isolation orbit-wars** : zéro partage venv ou code.
- **Submissions = CSV de probabilités** `[0,1]`, jamais classifications {0,1}.
- **Seed=42 partout** : np, catboost, random.
- **GroupKFold(Race+Year)** obligatoire — pas de naive KFold.
- **AUC tracked** par fold + OOF aggregate.
- **No LB grinding** : submit quand CV improvement > 0.5pp ET paired fold-by-fold positive (4 jours, 40 submits dispos, pas besoin de spammer).
- **Anti-AI-slop** : valeurs mesurées, pas "TBD" ni "expected".
- **No push origin sans demande explicite**.

## Workflow

```bash
# 1. Train + predict + CSV
.venv/Scripts/python.exe scripts/run_NN_<approach>.py

# 2. Submit
.venv/Scripts/kaggle.exe competitions submit playground-series-s6e5 \
  -f submissions/NN_<approach>.csv \
  -m "submit-NN: <approach>"

# 3. Update README tracking table + commit
```

## STANDING PROCEDURE T1-T5

| Trigger | Condition |
|---|---|
| T1 post-task-long | tâche > 30 min OU > 5 tool-calls OU 1+ commit |
| T2 fatigue ALERT | ≥1 indicateur ALERT |
| T3 fatigue STOP | ≥1 indicateur STOP |
| T4 milestone | submit Kaggle shippée OU script approche N complétée |
| T5 user-explicit | "memo update" / "session-end" / "STOP" / "pause" |

Invoque `.claude/skills/project-session-end/SKILL.md`.

## Pointers

- Memory : `C:/Users/pierr/.claude/projects/C--Dev-f1-pitstops-s6e5/memory/`
- Plan vivant : `README.md` (tracking table)
- Spec original (SUPERSEDED, archive only) : `docs/superpowers/specs/2026-05-26-f1-pitstops-tuto-design.md`
- Plan original (SUPERSEDED) : `docs/superpowers/plans/2026-05-26-f1-pitstops-tuto-plan.md`
- Deps research : `docs/research/2026-05-26-deps-versions.md`
- Kaggle : <https://www.kaggle.com/competitions/playground-series-s6e5>
