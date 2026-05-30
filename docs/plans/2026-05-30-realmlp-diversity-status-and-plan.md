# Status & forward plan — 2026-05-30 — RealMLP diversity push

> Durable record of the live plan (was only in conversation). Deadline **2026-05-31
> 23:59 UTC** (J-1). Mandate from user: **push score to max** (TabPFN + other levers),
> act autonomously on each verdict, probe-submit principled candidates (no LB grinding).

## Current standings (LB = the only trustworthy metric here)

| Submit | Model | OOF AUC | **Public LB** | Note |
|---|---|---|---|---|
| 01 | CatBoost defaults | 0.9294 | 0.94457 | anchor |
| 02 | killer-FE all-cats | 0.9794 | 0.94296 | CV mirage (regression) |
| 03 | Plan-2c CatBoost (drop 5 high-card crosses) | 0.95676 | 0.95083 | honest CatBoost |
| 07 | blend CatBoost+RealMLP w=0.73 | 0.95726 | 0.95269 | dragged down by CatBoost |
| 07b | blend w=0.50 | 0.95695 | 0.95365 | less CatBoost = better |
| **06** | **RealMLP-v1 ALONE** | 0.95409 | **0.95407** | **BEST so far** |

## THE key finding (LB-reversal, measured 2026-05-30)

**RealMLP alone (LB 0.95407) beats CatBoost Plan-2c (LB 0.95083) by +0.324pp — even
though the OOF said the opposite (RealMLP 0.954 < CatBoost 0.957).** CatBoost overfits
the random-row StratifiedKFold CV far more than RealMLP; on the true test the NN wins.
Blends are MONOTONE in CatBoost weight: w(CatBoost)=1.0→0.95083, 0.73→0.95269,
0.50→0.95365, 0.0→0.95407. **CatBoost should get ~0 weight.** RealMLP is the engine.

**Consequence:** the OOF is NOT a reliable guide for model/blend selection here. Use LB
probes (principled, ≤ a few) to decide. The winning blend is RealMLP + TabPFN (the two
strong non-GBDT families), CatBoost weight ~0.

## In-flight kernels (Kaggle T4, monitor armed)

- **RealMLP-v2** — slug `pguillemin/f1-s6e5-realmlp-v2`. Config improvement from targeted
  web research (PyTabKit docs): `ls_eps=0` ("label smoothing harms RealMLP for AUC") +
  `n_epochs=16` (yekenot's 5 << default 256, undertrained). `kernels/run_06_realmlp`.
- **TabPFN-v2** — slug `pguillemin/f1-s6e5-tabpfn` (kernel version 2). Fixed v1's
  `TabPFNLicenseError` by pinning `tabpfn==2.0.9` (TabPFN-v2 = Apache-2.0 weights, no
  license-token gate; TabPFN-3 default is gated, fails in batch). Context cap 100k/fold
  (research: TabPFN AUC rises to ~100k context). RAW features = max decorrelation.
  `kernels/run_08_tabpfn`.

## Forward plan (execute on kernel completion)

1. **RealMLP-v2 alone → probe LB.** Beats 0.95407? If yes, new best + final candidate.
2. **TabPFN-v2 alone → probe LB** (if it ran). How strong / how decorrelated?
3. **Blend the strong non-GBDT models** (best-RealMLP + TabPFN [+ tiny CatBoost only if it
   helps]). Weights chosen by LB-informed reasoning (NOT OOF — proven misleading), e.g.
   weight by relative LB strength or equal-weight of the two strongest; probe 1-2 variants.
   Isotonic-calibrate before blending (AUC-preserving, fixes per-year mean).
4. **Select 2 finals** before deadline. Current pick: slot A = best single/blend (≥0.95407),
   slot B = a decorrelated runner-up (RealMLP-v1 or the best blend). Never two near-identical.

## Tooling / artifacts

- Blend script: `scripts/run_07_isotonic_blend.py` (vectors-only, safe locally, no OOM).
  Ad-hoc weight blends done via inline python (see git log).
- Pulled outputs (local temp): `plan2c_out/`, `realmlp_out/` (= v1). v2/tabpfn to pull.
- **No local heavy training** (laptop OOM risk, only 3.7 GB free observed) — all model
  training on Kaggle. Local = only vector blends + submissions.

## Guardrails

seed=42 everywhere. No competition submit was blind — each probe answers a question.
No LB grinding (no fine weight sweeps on LB). LB is the only trustworthy metric (OOF
overfits via CatBoost CTR / random-row CV). Aadi external data HELPS (ablation measured
+0.078pp), keep it. GBDT+GBDT blend is a dead end (submit-04, rho=0.958).
