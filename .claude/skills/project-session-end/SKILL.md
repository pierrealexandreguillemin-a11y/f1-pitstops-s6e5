---
name: project-session-end
description: f1-pitstops-s6e5 STANDING PROCEDURE T1-T5 — 18-section autoportable session-end memo + MEMORY.md + history archive. Projected from C:/Dev/skills-templates/project-session-end/SKILL.md (greenfield profile, 19 markers customized).
---

> **Projected skill** — source canonique : `C:\Dev\skills-templates\project-session-end\SKILL.md`. Companion doctrine : `C:/Users/pierr/.claude/projects/C--Dev-f1-pitstops-s6e5/memory/feedback_session_end_template.md`.

# f1-pitstops-s6e5 — Project Session-End Memo Generator

Implements `f1-pitstops-s6e5` `CLAUDE.md` STANDING PROCEDURE NON-NÉGOCIABLE post-task memory + fatigue self-assessment. **18-section autoportable resume memo** for fresh-Claude reprise sans context conversationnel.

> Stack-aware : Python 3.13 + .venv dédié + Kaggle Playground tabular CSV submission, bench gates = paired CV-fold + LB delta vs baseline.

## When to invoke (T1-T5 triggers obligatoire)

| Trigger | Condition |
|---|---|
| **T1 post-task-long** | tâche > 30 min wall OU > 5 tool-calls OU 1+ commit shippé |
| **T2 fatigue ALERT** | ≥ 1 indicateur ALERT (per `feedback_post_task_procedure_and_fatigue.md`) |
| **T3 fatigue STOP** | ≥ 1 indicateur STOP |
| **T4 milestone naturelle** | notebook NN complété end-to-end OU submission Kaggle shippée OU décision archi (e.g. survival vs sequence en notebook 04) |
| **T5 user-explicit** | "memo update" / "doc de reprise" / "session-end" / "STOP" / "pause" / "fin de session" |

Self-trigger when planning a commit body — verify session-end memo updated BEFORE final user-facing message.

## What to do when invoked (11-step protocol)

### Step 1 — Read STANDING PROCEDURE source-of-truth

```bash
# Don't reinvent — read the canonical sources
cat C:/Dev/f1-pitstops-s6e5/CLAUDE.md                                                                     # §"STANDING PROCEDURE"
cat C:/Users/pierr/.claude/projects/C--Dev-f1-pitstops-s6e5/memory/feedback_session_end_template.md       # 18 sections + checklist autoportée
cat C:/Users/pierr/.claude/projects/C--Dev-f1-pitstops-s6e5/memory/feedback_post_task_procedure_and_fatigue.md   # 6-indicator fatigue rationale
```

### Step 2 — Sanity bash collect state (parallel)

```bash
cd C:/Dev/f1-pitstops-s6e5
git rev-parse HEAD
git log --oneline -10                                                              # recent commits
git status --short                                                                  # uncommitted
git rev-list --count origin/master..HEAD 2>/dev/null || echo "no remote"           # commits ahead
# N/A — Python pure, no build step
.venv/Scripts/python.exe -m pytest -q tests/ 2>&1 | tail -3
.venv/Scripts/python.exe -m ruff check . 2>&1 | tail -3
ls C:/Users/pierr/.claude/projects/C--Dev-f1-pitstops-s6e5/memory/MEMORY.md
.venv/Scripts/python.exe -m kaggle competitions submissions playground-series-s6e5 -v 2>&1 | head -5
```

### Step 3 — Determine session segment scope

- **Commits this segment** : count commits since session start (cumul user actions).
- **Files modified** : `git diff --stat origin/master..HEAD` (or `HEAD~N..HEAD` if no remote) to enumerate scope.
- **Tests delta** : compare current PASS count vs baseline at session start.
- **Bench/deploy artefacts created** : new files in `submissions/` + `data/processed/` cette session (e.g. `submissions/02_catboost_default.csv`, OOF `.npy`, cv `.csv`).
- **User-corrective-loops** : count instances of "user-flag → claude-fix" pattern (= candidate anti-patterns).

### Step 4 — Determine task tag

Concise kebab-case identifier for filename. Examples for f1-pitstops-s6e5 :
- `notebook-02-baseline-shipped`
- `notebook-03-domain-fe-cv-X.XXX`
- `submit-05-moe-bonus`
- `phase-0-setup-complete`
- `coxph-vs-deepsurv-decision`

### Step 5 — Generate filename

```
C:/Users/pierr/.claude/projects/C--Dev-f1-pitstops-s6e5/memory/project_session_end_<YYYY-MM-DD>-<task-tag>.md
```

### Step 6 — Build 18 sections per template (literal)

| § | Section | Content source |
|---|---|---|
| 1 | Resume protocol bash | literal copy-paste sanity checks (git + test runner + lint + paths runtime) |
| 2 | Session totals + HEAD | git state + tests + commits-ahead + push state |
| 3 | Deliverables (commits + files + LOC + bench/artefacts) | `git log` + `git diff --stat` + submissions CSV créés + OOF preds persistées |
| 4 | Hard signals fresh-Claude must respect | compose from CLAUDE.md projet + CLAUDE.md racine + feedbacks slow-changing — isolation from orbit-wars ; no LB grinding ; time-box 1.5x max per step ; seed fix everywhere ; group-aware CV mandatory ; calibration plot per submit ; anti-AI-slop |
| 5 | Next decision options A/B/C/D | + effort/coût estimé + ship-gate empirique + rollback path + recommendation |
| 6 | Tests/lint/quality status | pytest + ruff + pre-commit (auto on commit) + lock-sync (pre-push) |
| 7 | Runtime state (= source-of-truth ground-truth) | VALEURS ABSOLUES, pas de "voir tool externe". Kaggle submissions : sub_id + score + tags + COMPLETE/ERROR/PENDING ; local : last notebook executed + CV scores per fold + calibration Brier score |
| 8 | Mapping commit hash ↔ submission_id (anti faux-ami) | Pour CHAQUE commit récent cité comme "ship/baseline/feat" : Submitted to Kaggle? OUI/NON + sub_id si OUI. Si "Submitted=NON" pour commit cité comme ship → `⚠ commit-hash ≠ submission. Local-ship-only.` |
| 9 | Contrat ship explicit (anti-fuite-en-avant) | submit si CV improves baseline by >= 0.5pp ET paired fold-by-fold majority positive ET calibration not worse than baseline |
| 10 | Plans status DRIFT-check | docs/superpowers/specs/2026-05-26-f1-pitstops-tuto-design.md + docs/superpowers/plans/2026-05-26-f1-pitstops-tuto-plan.md. Checklist : status banner cohérent vs git log + MEMORY.md + spec block ; logbook append-only ; composite exit gate status connu |
| 11 | Anti-pattern counter unifié | pointer `feedback_diagnostic_first_doctrine.md` (au démarrage : aucun). NE PAS réécrire le compteur — citer current count + dernier ajout |
| 12 | Dead code post-pivot tracking | tout symbole landed dans une session où décision archi ultérieure a rendu le call-site caduc — listé avec décision (wire / docstring deprecation / revert) |
| 13 | Notebook 00-05 status (DONE/IN-PROGRESS/PENDING) (anti-prematuré-N+1) | Si Notebook N status=OPEN sans submit Kaggle shipped, ouvrir N+1 = anti-pattern doctrinal |
| 14 | Read order critical pour fresh-Claude | 10-20 docs ordonnés ("Si veut X → lire Y") : CLAUDE.md projet first, racine, memory feedbacks, plans actifs, code substrate à modifier next |
| 15 | Critique restante (TODO future) | unfinished items + estimated effort + push origin pending USER OK + plans synthèse REPORTÉ |
| 16 | OPEN QUESTION fresh-Claude | Q1 + Q2 + Q3 anti-décision-implicite (question 1-2 phrases + 2-3 hypothèses + recommandation par défaut + condition switch) |
| 17 | Fatigue self-assessment + 1-line user-facing | indicateurs OK/ALERT/STOP table (6 indicateurs per `feedback_post_task_procedure_and_fatigue.md`) |
| 18 | Instructions de reprise IDE-pasteable | bloc literal copy-paste depuis §1+§4+§5+§14+§16+§17. Auto-généré à chaque memo refresh. NE PAS générer ad-hoc dans chat — persister dans le memo |

### Step 7 — Verify 7 fresh-Claude-resume self-contained criteria

Per template "Checklist autoportée" :

1. **Sanity bash literal** : un fresh-Claude peut copier-coller §1 sans interprétation (commandes literally exécutables + paths absolus).
2. **State complete** : §2-§3 capture HEAD + commits + tests + runtime état sans ambiguïté.
3. **Hard signals enumerated** : §4 cite TOUS rules + invariants critiques (isolation orbit-wars, no LB grinding, time-box, seed=42, GroupKFold, calibration, anti-AI-slop).
4. **Next options actionnable** : §5 propose A/B/C/D avec coût + recommandation explicite.
5. **Read order ordered** : §14 priorise lecture pour fresh-Claude (canonical source first).
6. **OPEN QUESTION explicit** : §16 force user decision, NOT décision implicite Claude.
7. **IDE-pasteable bloc** : §18 contient TOUT le minimum pour fresh-Claude reprise (= condensé sections critiques).

Si < 7/7 → re-edit memo BEFORE write.

### Step 8 — Write memo file

`Write` tool to `C:/Users/pierr/.claude/projects/C--Dev-f1-pitstops-s6e5/memory/project_session_end_<DATE>-<task-tag>.md`.

### Step 9 — Update MEMORY.md entry 0 (= "CURRENT")

- Promote NEW memo as entry 0 "**CURRENT**".
- Demote previous entry 0 to "==== PRIOR SESSION ... — SUPERSEDED by 0 ====" (in place inside MEMORY.md).
- Keep entry 0 concise (~1 paragraph) — full detail is in memo file linked.
- Verify MEMORY.md size < 24.4 KB warning threshold (compress predecessor entries if needed).

### Step 10 — Update history archive

`C:/Users/pierr/.claude/projects/C--Dev-f1-pitstops-s6e5/memory/project_session_history_archive.md` : prepend current-date entry sous `## YYYY-MM-DD` header. One paragraph summary literal (= chronological full archive preserved across MEMORY.md compactions).

If file does not exist yet, create with header `# f1-pitstops-s6e5 session history archive` then prepend.

### Step 11 — Output user-facing recap

Format obligatoire :

```
Memory protocol complete. <N> fichiers memory updated/created cette session.

## Session-end memory bus complet
<table>

## Recap session <DATE> (1 paragraphe)
<HEAD + commits + key deliverables + next session prep>

## Next decision points (per memo §16)
- Q1 : ...
- Q2 : ...
- Q3 : ...

## Fatigue final
<STOP/ALERT/OK> — <indicateur principal>. Break clean obligatoire si STOP.

Reprise next session : copy-paste §18 IDE-pasteable du memo pour fresh-Claude. Protocol respecté.
```

## Anti-patterns (refuse to bless)

- **Skip section** : tous les 18 sections obligatoires, "moins important" = NO. Si une section n'a pas de contenu, écrire `N/A — <raison>`, jamais blank.
- **Section vide ou TBD** : remplir avec valeur réelle ou "N/A — <raison>" minimum, jamais blank ni "TBD" ni "expected" (cf. CLAUDE.md racine `feedback_benchmark_every_step`).
- **Fresh-Claude resume incomplet** : 7 self-contained criteria DOIVENT toutes PASS. Si < 7/7 → re-edit avant write.
- **MEMORY.md entry 0 size > 24.4 KB** : compresser predecessor entries simultanément.
- **Manual assembly bypass** : si user invoque "session-end" → INVOKE ce skill, ne pas réimplémenter manually (= drift risk inter-sessions).
- **Skip fatigue self-assessment** : §17 obligatoire avec 6 indicateurs table + 1-line user-facing.
- **Hardcode bench/deploy results** : §7 valeurs absolues, pas "voir bench/results/" — copier les valeurs DANS le memo.

## Cross-references

- `C:/Dev/f1-pitstops-s6e5/CLAUDE.md` `## STANDING PROCEDURE — POST-TASK MEMORY + FATIGUE SELF-ASSESSMENT` (= source-of-truth doctrinale projet).
- `C:/Dev/CLAUDE.md` racine (= rigueur + anti-hallucination + bench-every-step — applicable partout).
- `C:/Users/pierr/.claude/projects/C--Dev-f1-pitstops-s6e5/memory/feedback_session_end_template.md` (= template original 18 sections + checklist autoportée — projection locale du GENERIC).
- `C:/Users/pierr/.claude/projects/C--Dev-f1-pitstops-s6e5/memory/feedback_post_task_procedure_and_fatigue.md` (= STANDING PROCEDURE rationale + 6-indicator fatigue).
- `C:/Users/pierr/.claude/projects/C--Dev-f1-pitstops-s6e5/memory/feedback_per_task_memory_updates.md` (= per-task memo discipline).
- `C:/Users/pierr/.claude/projects/C--Dev-f1-pitstops-s6e5/memory/feedback_diagnostic_first_doctrine.md` (= anti-pattern counter source-of-truth).
- `C:/Users/pierr/.claude/projects/C--Dev-f1-pitstops-s6e5/memory/MEMORY.md` (= index target update entry 0).
- `C:/Users/pierr/.claude/projects/C--Dev-f1-pitstops-s6e5/memory/project_session_history_archive.md` (= chronological archive target).
- Project-specific cross-refs : tracking table in `README.md` (created Task 16) ; Kaggle competition page <https://www.kaggle.com/competitions/playground-series-s6e5> ; design spec `docs/superpowers/specs/2026-05-26-f1-pitstops-tuto-design.md` ; plan `docs/superpowers/plans/2026-05-26-f1-pitstops-tuto-plan.md`.
- Source canonique de la doctrine : `C:/Dev/feedback_session_end_template_GENERIC.md`.
- Source canonique du skill template : `C:/Dev/skills-templates/project-session-end/SKILL.md`.
- Skill orbit-wars de référence (premier projeté) : `C:/Dev/orbit-wars/.claude/skills/project-session-end/SKILL.md`.

## Outputs deliverables (literal report back)

After invocation completes, report :

1. Memo filename + LOC + sections count (= 18).
2. MEMORY.md entry 0 promotion verified.
3. Archive prepend verified.
4. 7 fresh-Claude self-contained criteria : `<X>/7` PASS.
5. Fatigue verdict + 1-line user-facing.
6. Total time spent (wall) — should be < 2 min.
