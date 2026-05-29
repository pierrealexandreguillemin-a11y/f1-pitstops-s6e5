# Finding — train/test split is RANDOM-ROW (validates StratifiedKFold + (Y,R,D) aggregates)

> **⚠️ CORRECTION (2026-05-29, post-LB).** The conclusion below — *"OOF ≈ LB,
> trustworthy, safe to stack"* — was **REFUTED by the leaderboard**: submit-02 OOF
> 0.9794 vs LB **0.94296** (3.6pp gap, regression vs submit-01 0.94457). The
> random-row split facts here are correct, BUT they do NOT make the OOF trustworthy:
> CatBoost's categorical CTR overfits the random-row CV (same cat levels in train+val
> folds). The honest signal is ~0.943 (numeric-only CatBoost). See
> `2026-05-29-submit-02-cv-lb-gap-postmortem.md`. Read that, not the conclusion below.

> **Date** : 2026-05-29. Item 3 of the submit-02 improvement inventory: verify the
> (Y,R,D) target-encoding aggregates are legitimate signal, not leakage, before
> trusting the ~0.978 OOF or stacking on it for submit-03.

## Method
Local analysis of `data/raw/{train,test}.csv` (439140 / 188165 rows). Compared
(Year,Race) and (Year,Race,Driver) membership, per-unit lap partitioning, and
within-unit target variance.

## Results

| Probe | Value |
|---|---|
| id overlap train∩test | 0 (disjoint rows) |
| (Year,Race) test ⊆ train | **100%** (same 104 races) |
| (Y,R,Driver) units test ⊆ train | **96.3%** |
| median laps / unit | train 11, test 5 (a unit's ~16 laps are split) |
| within-unit target mean (n≥10) | 0.204; all-0 units 29.9%, all-1 0.1% |
| test laps **interleaved** within train lap range | **96.2%** |
| test laps strictly after train (time split) | 3.8% |

## Interpretation

The competition split is **random-row within each (Y,R,D) unit**: the same races
and drivers appear in both train and test, and each unit's laps are randomly
partitioned (96.2% interleaved, not a time cut).

Consequences:
1. **The (Y,R,D) aggregates are legitimate, not leaky.** They encode a driver's
   pit frequency in a race (~0.2), computed for test from the unit's *train* laps
   (no test labels). Target varies within a unit (each has ~2 pit-laps of ~11), so
   the aggregate is signal, not a deterministic per-lap leak.
2. **OOF ≈ LB.** The OOF k-fold aggregate (a random subset of the unit's laps) is
   constructed the *same way* as the test aggregate → the ~0.978 OOF should track
   LB closely, with only a small optimism gap.
3. **StratifiedKFold is the correct CV here** — it matches the random-row test
   construction. submit-01's GroupKFold (whole races held out) was *pessimistic*
   (mismatched the split), which explains its +1.52pp CV-LB gap — **not** leakage.

## Bottom line
The 0.98-vs-0.944 discrepancy was never leakage: submit-01 used pessimistic
GroupKFold + a 14-feature model. submit-02 (99 feat, legitimate (Y,R,D) aggregates,
matched StratifiedKFold) should beat LB 0.944 substantially. Final arbiter remains
the leaderboard. Safe to trust the OOF for submit-03 stacking.

## Caveat
The ~3.8% time-structured units and the exact lap-selection mechanism are not fully
characterised; this is a strong but not exhaustive check. The LB will confirm.
