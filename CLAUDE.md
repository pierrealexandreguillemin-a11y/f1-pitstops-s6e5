# CLAUDE.md — `f1-pitstops-s6e5`

> **Scope** : ce fichier s'applique quand `cwd = C:/Dev/f1-pitstops-s6e5`. Charge en parallèle de `C:/Dev/CLAUDE.md` racine et de la memory dédiée `C:/Users/pierr/.claude/projects/C--Dev-f1-pitstops-s6e5/memory/`.

## Intent

Kaggle Playground Series S6E5 (Predicting F1 Pit Stops, deadline 2026-05-31, reward Swag). **Tuto-fun mode** — l'objectif est pédagogique, pas le leaderboard.

- Design spec : `docs/superpowers/specs/2026-05-26-f1-pitstops-tuto-design.md`
- Implementation plan : `docs/superpowers/plans/2026-05-26-f1-pitstops-tuto-plan.md`
- Deps research : `docs/research/2026-05-26-deps-versions.md`

## Hard signals (non-négociables)

- **Isolation orbit-wars** : zéro import, zéro venv partagé, zéro pression de temps partagée.
- **No LB grinding** : ~1 submit par notebook step, jamais plus. CV-LB gap = data point, pas une cible à chasser.
- **Time-box discipline** : step > 1.5× estimate → commit current state + move on.
- **Seed everything** : `np.random.seed(42)`, `torch.manual_seed(42)`, `catboost(random_seed=42)`. Reproducible.
- **Group-aware CV mandatory** : `GroupKFold(groups=Race+Year)` — jamais de naive KFold.
- **Calibration audit** : chaque submit produit un reliability diagram + Brier score.
- **Anti-AI-slop** : pas de "TBD", "expected", "should be". Que des valeurs mesurées.

## Stack — Python local-projet uniquement

- **Python 3.13.13** via `.venv/` local au projet (Python 3.11 initial du plan non installé sur la machine ; pivot 3.13 documenté `docs/research/2026-05-26-deps-versions.md`).
- **Zéro contamination** système ou autres projets : si tu touches au PATH global ou à une install Python hors `.venv/`, **stop**.
- `pyproject.toml` pins exacts ; `requirements.lock` capture résolus ; `scripts/check_lock_sync.py` vérifie la cohérence (pre-push hook).

### CONTAMINATION CONNUE — user-site Python 3.13

Le laptop a un user-site Python pollué : `C:/Users/pierr/AppData/Roaming/Python/Python313/Scripts/` contient ~30+ exes (`jupyter-nbconvert`, `detect-secrets`, `accelerate`, `celery`, `coverage`, `kaggle`, etc.) qui apparaissent dans `$PATH` AVANT `.venv/Scripts/`.

**Conséquence** : `.venv/Scripts/python.exe -m jupyter <cmd>` dispatche vers `jupyter-<cmd>.exe` via PATH lookup → trouve la version user-site (mauvais Python, dépendances absentes). Bug observé une fois sur `nbconvert` au bootstrap.

### RÈGLE DURE — Bash invocations

**Toujours préfixer** :

```bash
cd C:/Dev/f1-pitstops-s6e5 && source .venv/Scripts/activate && <command>
```

Après `source activate`, `which python | which jupyter | which kaggle | which pytest | which ruff` résolvent TOUS vers `.venv/Scripts/`. Vérifié 2026-05-27.

**Fallback équivalent** (si activate impossible) : appeler l'exe absolu directement, ex. `.venv/Scripts/jupyter-nbconvert.exe ...`, `.venv/Scripts/kaggle.exe ...`.

**À PROSCRIRE** : `python -m jupyter`, `python -m kaggle`, ou tout dispatcher qui sub-spawne via PATH sans activate préalable.

**Doctor** : `bash scripts/doctor.sh` audit ponctuel — vérifie qu'aucun tool ne résout hors venv.

## ISO/SOTA cadre (light, per spec §13)

- ISO/IEC 23053 lifecycle traceability (notebooks numérotés 00-05, commits taggés `submit-NN-<approach>`).
- ISO/IEC 25010 subset (functional + reliability + performance + maintainability + portability).
- SOTA ML rigor : seed=42 partout + GroupKFold(Race+Year) + OOF preds persistés + paired bench (`src/bench.py`) + calibration plot + SHAP (pas feature_importances_ seul).

## STANDING PROCEDURE — POST-TASK MEMORY + FATIGUE SELF-ASSESSMENT

Per `.claude/skills/project-session-end/SKILL.md` — triggers T1-T5 actifs :

| Trigger | Condition |
|---|---|
| **T1 post-task-long** | tâche > 30 min wall OU > 5 tool-calls OU 1+ commit shippé |
| **T2 fatigue ALERT** | ≥1 indicateur ALERT (per `memory/feedback_post_task_procedure_and_fatigue.md`) |
| **T3 fatigue STOP** | ≥1 indicateur STOP |
| **T4 milestone naturelle** | notebook NN complété end-to-end OU submission Kaggle shippée OU décision archi (e.g. survival vs sequence en notebook 04) |
| **T5 user-explicit** | "memo update" / "session-end" / "STOP" / "pause" / "fin de session" |

Invoque le skill `project-session-end` — auto-fills 18 sections from git/tests/kaggle state + writes memo + updates MEMORY.md + updates archive.

## Pointers

- Memory dir : `C:/Users/pierr/.claude/projects/C--Dev-f1-pitstops-s6e5/memory/`
- Design spec : `docs/superpowers/specs/2026-05-26-f1-pitstops-tuto-design.md`
- Implementation plan : `docs/superpowers/plans/2026-05-26-f1-pitstops-tuto-plan.md`
- Tracking table : `README.md` (créé Task 16)
- Skill session-end : `.claude/skills/project-session-end/SKILL.md` (projeté Task 10)
- Kaggle competition : <https://www.kaggle.com/competitions/playground-series-s6e5>
