# Post-mortem — submit-02 (Plan-2) CV-LB gap: the categorical-CTR mirage

> **Date** : 2026-05-29. submit-02 (Plan-2, "killer FE") shipped **OOF 0.9794** but
> **LB 0.94296**, a **regression** vs submit-01 (Plan-1) LB **0.94457**. A 3.6pp
> CV-LB gap. This documents the measured root cause and corrects two earlier wrong
> diagnoses of mine.

## Two diagnoses I made and MEASUREMENT killed

1. **"OOF ≈ LB, trustworthy" (pre-submit).** Refuted by the LB itself (0.943 vs 0.979).
   I had talked myself out of the leakage concern after the random-row split analysis.
2. **"Cross-fold target-encoding leak."** Refuted locally: leaky vs nested target
   encoding gave identical OOF (0.9305 vs 0.9306, LightGBM). Not the cause.

## The measured cause (LightGBM + CatBoost ablations, local, competition train only)

| Config | OOF AUC | Note |
|---|---|---|
| LightGBM base only | 0.9209 | per-lap numeric floor |
| LightGBM base + ALL submit-02 aggregates | 0.9318 | aggregates **generalize** (+0.01, ≈ LB) |
| LightGBM target-enc leaky vs nested | 0.9305 / 0.9306 | **no cross-fold leak** |
| CatBoost numeric-only (no cats), iter=800 | **0.9433** | **≈ LB 0.94296** — the honest signal |
| CatBoost + ALL cats, iter=800 | **0.9752** | **+3.2pp CV mirage** |
| CatBoost + cats minus 5 high-card crosses | 0.9524 | crosses = +2.3pp of the mirage |
| submit-02 CatBoost, ALL cats, 16000 iter | 0.9794 (OOF) / **0.94296 (LB)** | shipped |

**Root cause: CatBoost's categorical CTR (ordered target statistics + its internal
combinations up to `max_ctr_complexity=4`) overfits the random-row StratifiedKFold.**
Because the split is random-row, every categorical level / cross (e.g. `Driver_Race`,
`Race_Compound_Stint`) appears in both train and val folds, so the CTR memorizes
per-level target rates that look excellent in CV but add ~nothing on the held-out
test. The mirage is present **already at iter=800** → iterations were a minor
amplifier, not the cause. The 5 high-cardinality crosses contribute +2.3pp of it.

The explicit hand-built (Y,R,D) aggregates are NOT the problem: as numeric features
they give 0.9318 (LightGBM) / are absorbed into the 0.9433 numeric-only CatBoost,
and they generalize (≈ LB). It is specifically CatBoost's **native CTR on
high-cardinality categoricals** that inflates CV.

## The benchmark I documented and ignored

`defensive-pipeline-0.950-CV` (my own research doc) states a good model is **CV
0.947 (LightGBM) / 0.950 (CatBoost blend)**, per-year **0.909–0.934**, CatBoost at
**near-defaults**. submit-02 showed **0.979 / per-year 0.96–0.97 / 16000 iters** —
*above* that documented honest ceiling. A model that beats the best documented
pipeline by 3pp on CV is overfitting, not winning. I should have flagged it
pre-submission. The best documented single CatBoost (kernel 0.95259) caps at
**~8140 iters + auto_class_weights=Balanced + 2-seed**.

## Corrective direction

- **Numeric-only / low-card-cat CatBoost** is the honest base (~0.943–0.952 OOF).
  The path to *real* gain over submit-01 is dropping the high-card crosses
  (Plan-2c), NOT capping iterations alone (Plan-2b is insufficient — it keeps the
  crosses).
- **Any OOF must be validated against LB before it is trusted** (for stacking or
  selection). The ensemble plan stays FROZEN until a base beats 0.944 on LB.

## Lesson (feedback)
Measure before asserting a cause — I asserted two wrong causes before running the
ablation. And compare any result against the documented honest ceiling: a CV score
far above the best documented pipeline is a red flag, not a win.
