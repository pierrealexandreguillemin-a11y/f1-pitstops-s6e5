# Plan — submit-02 checkpoint/resume (timeout-proof relaunch)

> **Date** : 2026-05-29 (J-2, deadline 2026-05-31 23:59 UTC)
> **Context** : submit-02 kernel v1 (`22fc45c`) timed out architecture — 5 folds ×
> ~4.95 h = ~24.6 h CPU > 12 h Kaggle limit. All output written at the end →
> hard-kill at ~fold 2 → **zero artifacts**. This plan makes the kernel survive
> across multiple kernel runs with **no loss** and **no quality reduction**.

## Root cause (confirmed)

- Kaggle kills kernels hard at 12 h CPU / 12 h GPU. **No graceful shutdown.**
- `/kaggle/working` commits as kernel output **only on clean exit (code 0)**.
- v1 writes submission + parquet + metrics **after** the 5-fold loop (line 575+).
  Killed mid-loop → nothing committed. 12 h compute, 0 deliverable.

## Constraints (user, 2026-05-29)

1. **No quality reduction via iteration cut.** `iterations=11000` + early-stop=500
   stay untouched for the real model.
2. **Checkpoints for relaunch in another kernel without loss.**
3. **No GPU launch without a validated safety net** (would burn 30 h/week quota
   on a faulty kernel).

## Design

### Determinism = resume soundness
Steps 1–6 (load, domain FE, crosses, freq-encoding, `StratifiedKFold(seed=42,
shuffle)`, k-fold aggregates) are fully deterministic and seeded. Re-running them
in any kernel reproduces **byte-identical** folds. Therefore fold *N*'s training
data is identical regardless of which run trains it → a fold trained in run 1 can
be loaded in run 3. A startup assertion checks loaded `va_idx` == current split's
`va_idx` per fold (guards against any silent drift).

### Per-fold checkpoint (lightweight, ~2 MB/fold)
After each fold finishes training, **immediately** write:
- `ckpt/fold_{i}.npz` : `va_idx`, `va_pred`, `test_pred`, `feat_imp`
- `ckpt/fold_{i}_meta.json` : auc, brier, logloss, best_iter, seconds, per-year
- (optional) `ckpt/fold_{i}.cbm` model — kept in output for submit-03, **not**
  required for resume, **excluded** from the resume dataset to keep handoffs fast.

Resume needs only the npz+json (preds), so the cross-run upload stays ~10 MB.

### Time-budget guard (clean self-termination)
- Measured fold time : **4.95 h** (v1 logs, folds 0 & 1).
- `TIME_BUDGET_SECONDS = 11 * 3600` (1 h margin under 12 h).
- Adaptive estimate : prior = 5.5 h, updated to the mean of folds finished this run.
- Between folds: `if elapsed + estimate > BUDGET and folds remain → write
  RESUME_STATE.json, return cleanly`. Output commits. → **2 folds/run**.
- Guard is between folds only (CatBoost `fit` is one blocking call). 4.95 h/fold
  with an 11 h budget never starts a fold it can't finish; even a 5.5 h fold fits 2.

### Cross-run mechanism (dataset round-trip — proven)
Each run re-emits all prior ckpts into its own output (self-contained superset).
1. Run completes (COMPLETE) → `kaggle kernels output` (SDK path, Windows charmap-safe).
2. Stage `ckpt/*.npz` + `*_meta.json` → `dataset-metadata.json` →
   `kaggle datasets create | version` (helper `upload_ckpt.py`).
3. Next push: add ckpt dataset to `dataset_sources`. Kernel at startup globs
   `/kaggle/input/**/ckpt/fold_*.npz`, copies into `OUTPUT/ckpt/`, loads, skips.

Sequence: Run1 = folds 0–1 · Run2 = load 0–1 + train 2–3 · Run3 = load 0–3 +
train 4 + **assemble** (OOF, test = mean of 5, B0–B7 gates, submission, parquet).

### Completeness gate
- `< 5` folds present → write partial `fold_metrics.csv` + `RESUME_STATE.json`
  (completed folds, next action), print remaining, **return clean (no submission)**.
- `== 5` → full assembly + quality gates + all outputs (unchanged from v1).

### Smoke test = the safety net (throwaway, NOT the submission)
`SMOKE = True` module constant → `iterations=30`, runs in ~2 min. Validates the
full plumbing: train → save ckpt → clean exit → reload → skip → train next →
assemble → write submission. Proves resume works **before** committing real hours
or any GPU quota. Flipped back to `False` for real runs (visible in the diff).
Env vars can't be passed via `kaggle kernels push` → constant edit is the
correct toggle (kaggle-deployment skill).

### GPU (deferred, gated on smoke)
After smoke proves resume on CPU (0 quota): GPU becomes safe — a faulty run can't
burn quota for nothing because completed folds survive. CatBoost params use no
`rsm` → GPU-compatible (no pairwise-mode crash). CPU multi-run (0 quota, ~3 runs)
stays the fallback. CPU-vs-GPU decided **after** smoke, with the user (quota = user
resource).

## Files touched
- `kernels/run_02_killer_fe/run_02_killer_fe.py` — checkpoint/resume/budget logic.
- `kernels/run_02_killer_fe/kernel-metadata.json` — add ckpt dataset to
  `dataset_sources` for resume runs (run 2+).
- `kernels/run_02_killer_fe/upload_ckpt.py` — NEW, download-output + upload-dataset helper.

## Out of scope (flagged, separate)
- 0.98 OOF vs 0.944 LB gap: inspect train/test lap-overlap in the (Y,R,D)
  aggregates before spending a final submit slot. Does **not** block this fix.
